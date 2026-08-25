"""Agent policies: deterministic baselines + LLM agent with reliability wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .client import chat, parse_order
from . import client as client_mod
from .introspect import parse_introspection

_KB_PATH = Path(__file__).parent / "knowledge" / "PROMPT_KB.md"
_KB_CACHE: str | None = None


def _kb_text() -> str:
    """Load the KB once; returns '' if missing (graceful degradation)."""
    global _KB_CACHE
    if _KB_CACHE is None:
        try:
            _KB_CACHE = _KB_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            _KB_CACHE = ""
    return _KB_CACHE


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
    tag: str | None = None                 # result-file tag (e.g. per model)
    # --- Wave 2: introspection + knowledge base ---
    introspect: bool = False               # emit ORDER/CONFIDENCE/REASONING, log traces
    kb: bool = False                       # inject the decision playbook into the prompt
    kb_placement: str = "inline"           # "inline" (user msg) | "system" (system msg) | "pointer" (guide only)
    conf_threshold: float | None = None    # below this, fall back to anchor (self-gate)


class LLMAgent:
    def __init__(self, role: str, cfg: LLMAgentConfig | None = None):
        self.role = role
        self.cfg = cfg or LLMAgentConfig()
        self.anchor = OrderUpToAgent() if self.cfg.anchor_margin is not None else None
        self.failures = 0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.traces: list[dict] = []   # Wave 2: per-decision {order, confidence, reasoning}
        self.last_api_reasoning: list | None = None   # reasoning_content from the API (if any)

    def decide(self, ctx: dict) -> int | None:
        use_system = self.cfg.kb and self.cfg.kb_placement == "system"
        use_pointer = self.cfg.kb and self.cfg.kb_placement == "pointer"
        if use_system:
            # KB + role live in the system message; state + question in the user message.
            system = build_system_prompt(self.role, self.cfg.prompt_variant,
                                         kb=True, introspect=self.cfg.introspect)
            user = build_user_prompt(self.role, ctx, self.cfg.prompt_variant,
                                     introspect=self.cfg.introspect)
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
        else:
            prompt = build_prompt(self.role, ctx, self.cfg.prompt_variant,
                                  introspect=self.cfg.introspect,
                                  kb=self.cfg.kb, kb_pointer=use_pointer)
            messages = [{"role": "user", "content": prompt}]
        self.calls += 1
        samples = chat(
            messages,
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            n=self.cfg.voting,
        )
        # token accounting + reasoning trace capture
        usage = getattr(client_mod, "last_usage", {}) if client_mod else {}
        self.prompt_tokens += usage.get("prompt", 0)
        self.completion_tokens += usage.get("completion", 0)
        reasons = getattr(client_mod, "last_reasoning", None) if client_mod else None
        self.last_api_reasoning = reasons or [None] * len(samples)
        if self.cfg.introspect:
            return self._decide_introspect(ctx, samples)
        orders = [parse_order(s) for s in samples]
        parsed = [o for o in orders if o is not None]
        if not parsed:
            self.failures += 1
            return self._fallback(ctx)
        order = median(parsed) if self.cfg.voting > 1 else parsed[0]
        return self._apply_wrappers(ctx, int(order))

    def _decide_introspect(self, ctx: dict, samples: list[str]) -> int | None:
        """Parse ORDER/CONFIDENCE/REASONING/THINKING; self-gate on low confidence."""
        parsed = []
        for s in samples:
            p = parse_introspection(s)
            if p is not None:
                parsed.append(p)
        if not parsed:
            # format failure -> count as failure + fall back (keeps metric honest)
            self.failures += 1
            return self._fallback(ctx)
        # pick the median order across samples (self-consistency, as voting)
        best = sorted(parsed, key=lambda p: p["order"])[len(parsed) // 2]
        conf = best.get("confidence")
        order = best["order"]
        api_reason = (self.last_api_reasoning or [None])[0] if self.last_api_reasoning else None
        # self-gate: low confidence -> fall back to the safe deterministic anchor
        if conf is not None and self.cfg.conf_threshold is not None and conf < self.cfg.conf_threshold:
            safe = self.anchor(ctx) if self.anchor is not None else self._fallback(ctx)
            self.traces.append({"ctx": ctx, "order": order, "confidence": conf,
                                "reasoning": best.get("reasoning", ""),
                                "thinking": best.get("thinking", ""),
                                "api_reasoning": api_reason, "gated": True,
                                "order_used": int(safe)})
            return self._apply_wrappers(ctx, int(safe))
        self.traces.append({"ctx": ctx, "order": order, "confidence": conf,
                            "reasoning": best.get("reasoning", ""),
                            "thinking": best.get("thinking", ""),
                            "api_reasoning": api_reason, "gated": False,
                            "order_used": int(order)})
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


def build_prompt(role: str, ctx: dict, variant: str = "default",
                 introspect: bool = False, kb: bool = False,
                 kb_pointer: bool = False) -> str:
    """Inline (user-message) prompt: role + state + (optional KB) + question.

    kb=True with kb_pointer=False injects the playbook verbatim.
    kb_pointer=True only *points* to the playbook and asks the model to consult
    its own internalized knowledge / self-check steps — testing whether
    self-directed rule application beats injected context.
    """
    goal = (
        "minimize the weighted average of backlog and holding costs"
        if variant == "weighted"
        else "minimize total supply chain cost"
    )
    p = (
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
    )
    if kb_pointer:
        p += (
            "=== DECISION PLAYBOOK (knowledge base) ===\n"
            "There is a decision playbook of supply-chain rules you should follow.\n"
            "You do not see its contents here: use your own knowledge of inventory "
            "management and beer-game best practices, and self-check your reasoning "
            "against these principles before answering: don't over-react to demand "
            "steps, cover backlog before adding inventory, account for the pipeline, "
            "smooth noisy demand, and prefer a conservative order when uncertain.\n"
            "=== END PLAYBOOK POINTER ===\n\n"
        )
    elif kb:
        kb = _kb_text()
        if kb:
            p += "=== DECISION PLAYBOOK (knowledge base) ===\n"
            p += "Use these rules when they apply. They are ground truth for this game:\n\n"
            p += kb + "\n\n=== END PLAYBOOK ===\n\n"
    if introspect:
        p += (
            "Before answering, think step by step about your state and which principle "
            "applies. Then answer in EXACTLY this 4-line format — you MUST include all "
            "four lines, in this order, nothing else:\n"
            "THINKING: <one sentence: the key fact + which principle applies>\n"
            "ORDER: <integer, the number of units to order this week, 0 or more>\n"
            "CONFIDENCE: <0.0 to 1.0, how sure you are>\n"
            "REASONING: <one short sentence: the key fact + which principle you applied>\n"
        )
    else:
        p += "Reply with ONLY an integer: the number of units to order this week (0 or more)."
    return p


def build_system_prompt(role: str, variant: str = "default", kb: bool = False,
                        introspect: bool = False) -> str:
    """System-message prompt: identity + goal + (KB) — the persistent instruction layer.

    Kept separate from the per-week state (which goes in the user message) so we
    can test whether separating rules-from-data changes how the model handles them.
    """
    goal = (
        "minimize the weighted average of backlog and holding costs"
        if variant == "weighted"
        else "minimize total supply chain cost"
    )
    p = (
        f"You are the {role.upper()} in a four-stage beer supply chain:\n"
        "Retailer -> Wholesaler -> Distributor -> Factory.\n"
        "You manage inventory at your stage. Each week you decide how many units to order "
        f"from your supplier (upstream). Goal: {goal}.\n"
        "Holding cost is $1 per unit per week; backlog cost is $2 per unit per week.\n"
        "Shipments from your supplier take 2 weeks to arrive.\n\n"
    )
    if kb:
        kb = _kb_text()
        if kb:
            p += "=== DECISION PLAYBOOK (knowledge base) ===\n"
            p += "These rules are ground truth for this game. Apply them when they match "
            p += "your state each week:\n\n"
            p += kb + "\n\n=== END PLAYBOOK ===\n"
    if introspect:
        p += (
            "\nAnswer in EXACTLY this 3-line format (nothing else):\n"
            "ORDER: <integer, the number of units to order this week, 0 or more>\n"
            "CONFIDENCE: <0.0 to 1.0, how sure you are>\n"
            "REASONING: <one short sentence: the key fact + which playbook rule you applied>\n"
        )
    return p


def build_user_prompt(role: str, ctx: dict, variant: str = "default",
                      introspect: bool = False) -> str:
    """User-message prompt for the system-placement variant: just the state + question."""
    p = (
        f"Current week: {ctx['t']}\n"
        f"Your state at the start of the week:\n"
        f"- On-hand inventory: {ctx['on_hand']}\n"
        f"- Backlog (unfilled orders owed to your customer): {ctx['backlog']}\n"
        f"- Outstanding orders (ordered but not yet received): {ctx['outstanding']}\n"
        f"- Incoming order from your customer last week: {ctx['incoming_last']}\n"
        f"- Incoming order from your customer this week: {ctx['incoming_now']}\n"
        f"- Your recent orders: {ctx['last_orders']}\n\n"
    )
    if introspect:
        p += "What is your ORDER, CONFIDENCE, and REASONING this week?"
    else:
        p += "How many units do you order this week? Reply with ONLY an integer."
    return p
