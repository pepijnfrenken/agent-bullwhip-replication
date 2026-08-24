"""Agent policies: deterministic baselines + LLM agent with reliability wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .client import chat, parse_order


# --------------------------------------------------------------------------- #
# Deterministic baselines
# --------------------------------------------------------------------------- #
class MirrorAgent:
    """Order what your customer ordered last week (proportional-to-incoming)."""

    def __call__(self, ctx: dict) -> int:
        return int(ctx["incoming_last"])


class OrderUpToAgent:
    """Order-up-to with exponential-smoothing forecast (paper eqs. 3-5).

    q_t = max(0, theta * q_hat_t - IP_t),  q_hat_t = lam * q_{k-1,t-1} + (1-lam) * q_hat_{t-1}
    theta default = lead time + 1 (protection interval), lam default 0.5.
    """

    def __init__(self, theta: float = 3.0, lam: float = 0.5):
        self.theta = theta
        self.lam = lam
        self._forecast = 0.0

    def __call__(self, ctx: dict) -> int:
        last_incoming = ctx["incoming_last"]
        self._forecast = self.lam * last_incoming + (1 - self.lam) * self._forecast
        ip = ctx["on_hand"] + ctx["outstanding"] - ctx["backlog"]
        return int(max(0, round(self.theta * self._forecast - ip)))


# --------------------------------------------------------------------------- #
# LLM agent with reliability wrappers
# --------------------------------------------------------------------------- #
@dataclass
class LLMAgentConfig:
    voting: int = 1            # samples per decision; median when >1 (self-consistency)
    guardrail_cap: int | None = None       # hard cap on order quantity
    guardrail_ratio: float | None = None   # cap = max(cap, ratio * incoming_last)
    anchor_margin: int | None = None       # clamp order within +/-margin of ES anchor
    temperature: float = 0.7
    prompt_variant: str = "default"        # "default" | "weighted" (paper reframe)
    model: str | None = None
    fallback: str = "mirror"               # what to do on parse failure: mirror | anchor | zero


class LLMAgent:
    def __init__(self, role: str, cfg: LLMAgentConfig | None = None):
        self.role = role
        self.cfg = cfg or LLMAgentConfig()
        self.anchor = OrderUpToAgent() if self.cfg.anchor_margin is not None else None
        self.failures = 0
        self.calls = 0

    def decide(self, ctx: dict) -> int | None:
        prompt = build_prompt(self.role, ctx, self.cfg.prompt_variant)
        self.calls += 1
        samples = chat(
            [{"role": "user", "content": prompt}],
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            n=self.cfg.voting,
        )
        orders = [parse_order(s) for s in samples]
        parsed = [o for o in orders if o is not None]
        if not parsed:
            self.failures += 1
            return self._fallback(ctx)
        order = median(parsed) if self.cfg.voting > 1 else parsed[0]
        return self._apply_wrappers(ctx, int(order))

    def _fallback(self, ctx: dict) -> int | None:
        if self.cfg.fallback == "mirror":
            return int(ctx["incoming_last"])
        if self.cfg.fallback == "zero":
            return 0
        if self.cfg.fallback == "anchor" and self.anchor is not None:
            return self.anchor(ctx)
        return int(ctx["incoming_last"])

    def _apply_wrappers(self, ctx: dict, order: int) -> int:
        cap = self.cfg.guardrail_cap
        if self.cfg.guardrail_ratio is not None:
            cap = max(cap or 0, int(self.cfg.guardrail_ratio * ctx["incoming_last"]))
        if cap is not None:
            order = max(0, min(order, cap))
        if self.cfg.anchor_margin is not None and self.anchor is not None:
            base = self.anchor(ctx)
            order = max(0, min(order, base + self.cfg.anchor_margin))
            order = max(order, base - self.cfg.anchor_margin) if base - self.cfg.anchor_margin > 0 else max(order, 0)
        return int(order)

    def __call__(self, ctx: dict) -> int:
        q = self.decide(ctx)
        return int(max(0, q)) if q is not None else 0


def build_prompt(role: str, ctx: dict, variant: str = "default") -> str:
    goal = (
        "minimize the weighted average of backlog and holding costs"
        if variant == "weighted"
        else "minimize total supply chain cost"
    )
    return (
        f"You are the {role.upper()} in a four-stage beer supply chain:\n"
        "Retailer -> Wholesaler -> Distributor -> Factory.\n"
        "You manage inventory at your stage. Each week you decide how many units to order "
        f"from your supplier (upstream). Goal: {goal}.\n\n"
        f"Current week: {ctx['t']}\n"
        f"Your state at the start of the week:\n"
        f"- On-hand inventory: {ctx['on_hand']}\n"
        f"- Backlog (unfilled orders owed to your customer): {ctx['backlog']}\n"
        f"- Outstanding orders (ordered but not yet received): {ctx['outstanding']}\n"
        f"- Incoming order from your customer last week: {ctx['incoming_last']}\n"
        f"- Incoming order from your customer this week: {ctx['incoming_now']}\n"
        f"- Your recent orders: {ctx['last_orders']}\n\n"
        "Holding cost is $1 per unit per week; backlog cost is $2 per unit per week.\n"
        "Shipments from your supplier take 2 weeks to arrive.\n\n"
        "Reply with ONLY an integer: the number of units to order this week (0 or more)."
    )
