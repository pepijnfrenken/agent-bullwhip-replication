"""LLM-as-Configurator: the LLM sets the policy ONCE, a deterministic rule executes.

Wave 3 idea #1 (from study-design doc): instead of the LLM deciding every weekly
order (144 calls/game), the LLM chooses inventory-policy parameters once at the
start (theta/lambda/guardrail), then a deterministic order-up-to rule with ES
smoothing runs the 36 weeks. Cost: ~1 LLM call per game vs 144.

This is the "AI as configurator, not operator" play — the cheapest reliable
autonomous supply chain. The question it answers: can a single LLM judgment
call (params) + a classical rule match or beat 144 LLM decisions, at 1/144th
the inference cost and near-zero stochasticity (reliability)?
"""
from __future__ import annotations

import json
import re

from .agents import OrderUpToAgent
from .client import chat, parse_order

_PARAM_RE = {
    "theta": re.compile(r"theta\D{0,12}(\d+(?:\.\d+)?)", re.I),
    "lambda": re.compile(r"lambda\D{0,12}(\d+(?:\.\d+)?)", re.I),
    "cap": re.compile(r"cap\D{0,12}(\d+(?:\.\d+)?)", re.I),
}


def parse_params(text: str) -> dict | None:
    """Extract theta/lambda/cap from a configurator response. None if unparseable."""
    if not text:
        return None
    out: dict = {}
    for key, rx in _PARAM_RE.items():
        m = rx.search(text)
        if m:
            out[key] = float(m.group(1))
    if "theta" not in out:
        return None
    out.setdefault("lambda", 0.5)
    return out


class ConfiguratorAgent:
    """Deterministic order-up-to rule, with params chosen once by an LLM.

    The LLM sees the demand pattern summary (or nothing, if `blind`) and picks
    theta (protection-interval multiplier), lambda (smoothing), and an optional
    order cap. The rule then executes every week with zero stochasticity.
    """

    def __init__(self, role: str, model: str | None = None, blind: bool = False,
                 fallback: tuple[float, float, float | None] = (3.0, 0.5, None)):
        self.role = role
        self.model = model
        self.blind = blind
        self.fallback = fallback
        self.rule = OrderUpToAgent()  # placeholder; set by configure()
        self._cap: float | None = None
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.params: dict | None = None

    def configure(self, demand_summary: str = "") -> None:
        """One LLM call per game: choose theta/lambda/cap for this role."""
        prompt = (
            f"You are configuring an inventory agent for the {self.role.upper()} stage of a "
            "4-echelon beer supply chain (Retailer -> Wholesaler -> Distributor -> Factory).\n"
            "The agent will place a weekly order using an order-up-to rule with exponential "
            "smoothing: q_t = max(0, theta * forecast_t - IP_t), where forecast updates as "
            "forecast_t = lambda * incoming_{t-1} + (1-lambda) * forecast_{t-1}.\n"
            "You must choose good parameter values to minimize total holding + backlog cost "
            "($1/unit/week holding, $2/unit/week backlog; shipments take 2 weeks).\n"
            f"Demand context: {demand_summary if demand_summary else 'not provided — use your knowledge of the beer game.'}\n"
            "Reply with ONLY a compact JSON object, e.g.: {\"theta\": 3.0, \"lambda\": 0.5, \"cap\": 20}\n"
            "theta should be roughly the lead time + 1 (protection interval); cap is an optional "
            "maximum order quantity (omit or set null for none)."
        )
        from . import client as client_mod
        samples = chat([{"role": "user", "content": prompt}], model=self.model, max_tokens=80)
        usage = getattr(client_mod, "last_usage", {})
        self.prompt_tokens += usage.get("prompt", 0)
        self.completion_tokens += usage.get("completion", 0)
        params = None
        for s in samples:
            params = parse_params(s)
            if params:
                break
        if params is None:
            self.failures += 1
            theta, lam, cap = self.fallback
            params = {"theta": theta, "lambda": lam, "cap": cap}
        self.params = params
        self.rule = OrderUpToAgent(theta=params.get("theta", 3.0), lam=params.get("lambda", 0.5))
        self._cap = params.get("cap")

    def __call__(self, ctx: dict) -> int:
        q = self.rule(ctx)
        if self._cap is not None:
            q = min(q, int(self._cap))
        return int(max(0, q))


def run_configurator_game(model: str, demand: list[int], horizon: int,
                          blind: bool = False, demand_summary: str = "") -> dict:
    """Configure once per role, then run the deterministic game. Returns log + tokens."""
    from .engine import ROLES, SimConfig, run_game
    agents = {}
    for r in ROLES:
        a = ConfiguratorAgent(r, model=model, blind=blind)
        a.configure(demand_summary)
        agents[r] = a
    log = run_game(agents, demand, SimConfig(horizon=horizon))
    return {
        "log": log,
        "params": {r: agents[r].params for r in ROLES},
        "failures": {r: agents[r].failures for r in ROLES},
        "prompt_tokens": sum(a.prompt_tokens for a in agents.values()),
        "completion_tokens": sum(a.completion_tokens for a in agents.values()),
        "total_cost": log.total_cost(),
    }
