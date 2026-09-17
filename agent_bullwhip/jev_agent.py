"""Jev (TypeSafe) agent — System One decisions for the Beer Game.

Why this exists: the replication's surviving claim is that a deterministic
order-up-to wrapper beats an LLM agent, and that the LLM *inside* the wrapper
costs +12-34%. Its dead claim #1 was the confidence self-gate: the LLM's
introspected confidence fired on 0.2% of decisions and was inert. TypeSafe's
Jev is a System One model: it returns typed answers (choice/score/noul) with a
probability distribution and a confidence derived from that distribution. So it
tests three things the text LLM could not:

  1. a *calibrated* confidence signal -> the gate has something to gate on;
  2. no text generation -> no parse failures, no mirror-fallback laundering
     (the verbal config's hidden 21-29% silent fallbacks);
  3. atomic questions composed in code -> the model supplies only the judgment,
     the arithmetic stays deterministic (the project's own thesis, run forward).

Modes (all single-request, parallel questions):
  choice_grid  : Choice over explicit order quantities (0..grid_max step grid_step)
  choice_mult  : Choice over coverage multipliers kappa on the ES forecast;
                 order = max(0, round(kappa * q_hat - IP)). The model picks
                 *how much to cover*, the formula does the arithmetic.
  expectation  : same question, but order = sum(p_i * value_i) over the grid /
                 multipliers (uses the full distribution, not just the argmax)
  reads        : Noul/Score state reads (below target? backlog severity? demand
                 rising?) composed into an order in code with the anchor.

All modes can be confidence-gated: confidence < conf_threshold -> anchor.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.local"
API_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
DEFAULT_MODEL = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")

SITUATION = (
    "MIT Beer Game, 4-echelon supply chain (retailer -> wholesaler -> distributor -> factory). "
    "Each week you place ONE order with your supplier; units arrive after a 2-week lead time. "
    "Costs: $1 per unit held per week, $2 per unit of unfilled backlog per week. Lower total cost is better. "
    "You know only your own on-hand stock, backlog, outstanding pipeline orders, and the orders "
    "your customer placed (last week's everywhere; this week's only for the retailer). "
    "Orders of 0 with a backlog, or large orders after an order spike, both amplify the bullwhip effect."
)


def _load_key() -> str | None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def ask(state, questions: dict, model: str | None = None, timeout: float = 30.0,
        retries: int = 3) -> dict:
    """One POST to /v1/systemone. Retries 429/529 with exponential backoff."""
    key = _load_key()
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY not set — put it in the environment or in "
            f"{ENV_FILE} as TYPESAFE_API_KEY=... (dashboard: console.typesafe.ai)")
    payload = {"state": state, "model": model or DEFAULT_MODEL, "questions": questions}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=timeout,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"})
            if r.status_code in (429, 529):
                raise requests.HTTPError(f"HTTP {r.status_code} (rate/overload)")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — retry anything transient
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"TypeSafe call failed after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# question builders
# --------------------------------------------------------------------------- #
def choice(instructions: str, options: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #
@dataclass
class JevAgentConfig:
    mode: str = "choice_mult"        # choice_grid | choice_mult | expectation | reads
    grid_max: int = 40               # choice_grid top option
    grid_step: int = 2               # choice_grid spacing
    warm_start: bool = True          # seed the ES forecast from last_orders (mid-game probe states)
    mult_opts: tuple[float, ...] = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    theta: float = 3.0               # ES anchor protection interval
    lam: float = 0.5                 # ES anchor smoothing
    conf_threshold: float | None = None   # below -> anchor (calibrated self-gate)
    anchor_margin: int | None = None      # clamp to +/- margin of the anchor
    jitter_seed: int = 0                  # mode="jitter_only": seeds the pseudo-random perturbation
    jitter_span: int = 6                  # mode="jitter_only": fallback span if no anchor_margin
    fallback: str = "anchor"         # anchor | mirror | zero  (on any failure)
    include_reads: bool = True       # ask the auxiliary read questions too (costs ~nothing)
    model: str | None = None
    tag: str | None = None
    timeout: float = 30.0


class JevAgent:
    """Same call interface as LLMAgent: agent(ctx) -> int order."""

    def __init__(self, role: str, cfg: JevAgentConfig | None = None):
        self.role = role
        self.cfg = cfg or JevAgentConfig()
        self._forecast = 0.0
        self._forecast_t: object = object()   # week the forecast was last updated for
        self._warmed = False
        self.calls = 0
        self.failures = 0
        self.gate_fires = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.traces: list[dict] = []
        self.last_decision_meta: dict = {}

    # -- deterministic components ------------------------------------------- #
    def es_forecast(self, ctx: dict) -> float:
        """Exponential smoothing, updated AT MOST ONCE per game week.

        The anchor and the composition both need q_hat; without this guard the
        recursion stepped twice per decision (and up to three times when the
        confidence gate or the margin clamp fired), which silently drifted the
        anchor. Idempotent per `t`.
        """
        t = ctx.get("t")
        if t is not None and t == self._forecast_t:
            return self._forecast
        if self.cfg.warm_start and not self._warmed and ctx.get("last_orders"):
            # Probe states are mid-game snapshots, so a cold q_hat=0 is not representative;
            # replay the known order history through the same recursion the game would have.
            for q in ctx["last_orders"]:
                self._forecast = self.cfg.lam * q + (1 - self.cfg.lam) * self._forecast
            self._warmed = True
        self._forecast = self.cfg.lam * ctx.get("incoming_last", 0) + (1 - self.cfg.lam) * self._forecast
        self._forecast_t = t
        return self._forecast

    def anchor(self, ctx: dict) -> int:
        ip = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
        return int(max(0, round(self.cfg.theta * self.es_forecast(ctx) - ip)))

    def _fallback(self, ctx: dict) -> int:
        if self.cfg.fallback == "mirror":
            return int(ctx.get("incoming_last", 0))
        if self.cfg.fallback == "zero":
            return 0
        return self.anchor(ctx)

    def _state(self, ctx: dict) -> dict:
        seen = {k: ctx.get(k) for k in ("t", "on_hand", "backlog", "outstanding",
                                        "incoming_last", "incoming_now", "last_orders") if k in ctx}
        return {"situation": SITUATION, "role": self.role, "state": seen}

    def _questions(self) -> tuple[dict, dict]:
        """Returns (questions, meta) — meta says how to compose the answer."""
        q: dict = {}
        meta: dict = {}
        if self.cfg.mode == "choice_grid":
            opts = {str(x): (f"Order {x} units" + (" (nothing)" if x == 0 else ""))
                    for x in range(0, self.cfg.grid_max + 1, self.cfg.grid_step)}
            q["order"] = choice(
                "How many units should you order from your supplier this week? "
                "Weigh holding cost ($1/unit/week) against backlog cost ($2/unit/week), "
                "and avoid amplifying demand swings.", opts)
            meta["compose"] = "grid"
        elif self.cfg.mode in ("choice_mult", "expectation"):
            opts = {}
            for k in self.cfg.mult_opts:
                opts[str(k)] = (
                    "Order nothing" if k == 0 else
                    f"Set stock position to {k:g}x the smoothed demand forecast "
                    f"(k={k:g}); k<1 under-covers, k=1 covers the forecast, k>1 builds buffer")
            q["coverage"] = choice(
                "Choose the coverage multiplier for your order-up-to decision: your order "
                "becomes max(0, k * smoothed_forecast - stock_position), where stock_position = "
                "on_hand + pipeline - backlog and smoothed_forecast is exponential smoothing "
                "(lambda=0.5) of your customer's orders. Pick the multiplier that best "
                "controls total cost without amplifying the bullwhip.", opts)
            meta["compose"] = "mult"
        elif self.cfg.mode == "reads":
            q["below_target"] = noul(
                "Is your stock position (on_hand + pipeline - backlog) below the level needed to "
                "cover the next two weeks of demand?",
                true="Stock position is short of ~2 weeks of demand", false="Stock position covers it")
            q["backlog_pressure"] = score(
                "How much pressure does the current backlog put on your next order?",
                ["No backlog; nothing to recover",
                 "Small backlog; recover gradually",
                 "Large backlog; must order aggressively to recover"])
            q["demand_trend"] = score(
                "How is your customer's order volume trending?",
                ["Falling", "Flat", "Rising"])
            meta["compose"] = "reads"
        if self.cfg.include_reads and self.cfg.mode in ("choice_grid", "choice_mult", "expectation"):
            # near-free parallel reads, logged for the audit trail (speculative fan-out)
            q["below_target"] = noul(
                "Is your stock position (on_hand + pipeline - backlog) below the level needed to "
                "cover the next two weeks of demand?")
            q["backlog_pressure"] = score(
                "How much pressure does the current backlog put on your next order?",
                ["No backlog; nothing to recover",
                 "Small backlog; recover gradually",
                 "Large backlog; must order aggressively to recover"])
        return q, meta

    def __call__(self, ctx: dict) -> int:
        """The engine calls `agents[r](ctx)` — same interface as the other agents."""
        return self.decide(ctx)

    # -- the decision ------------------------------------------------------- #
    def decide(self, ctx: dict) -> int:
        if self.cfg.mode == "anchor_only":
            # The wrapper ablation: the deterministic anchor with NO API calls.
            # Isolates how much of a gated config's score is the model vs the formula
            # (and how much of the score is this agent's warm-started forecast).
            order = self.anchor(ctx)
            self.last_decision_meta = {"is_fallback": False, "order": order,
                                       "confidence": None, "gate_fired": False,
                                       "detail": {"mode": "anchor_only"}}
            return order
        if self.cfg.mode == "jitter_only":
            # The decisive control for "does the model have judgment?": apply the SAME
            # bounded perturbation the model's ungated calls apply (±margin), but from a
            # seeded RNG instead of the model. Same clamp, no API calls. If random jitter
            # tracks the model's jitter in cost, the model added no information.
            import hashlib
            a = self.anchor(ctx)
            key = f"{self.cfg.jitter_seed}:{ctx.get('role')}:{ctx.get('t')}".encode()
            h = int(hashlib.sha256(key).hexdigest()[:8], 16)
            span = self.cfg.anchor_margin if self.cfg.anchor_margin is not None else self.cfg.jitter_span
            delta = (h % (2 * span + 1)) - span
            order = int(max(0, a + delta))
            self.last_decision_meta = {"is_fallback": False, "order": order, "confidence": None,
                                       "gate_fired": False, "anchor": a,
                                       "detail": {"mode": "jitter_only", "delta": delta}}
            return order
        q, meta = self._questions()
        self.calls += 1
        try:
            resp = ask(self._state(ctx), q, model=self.cfg.model, timeout=self.cfg.timeout)
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            self.last_decision_meta = {"is_fallback": True, "fallback_reason": f"api_error: {e}"}
            return self._fallback(ctx)

        usage = resp.get("usage") or {}
        self.prompt_tokens += int(usage.get("input_tokens", 0))
        self.completion_tokens += int(usage.get("output_tokens", 0))
        ans = resp.get("answers") or {}
        order, conf, detail = self._compose(ctx, ans, meta)
        if order is None:
            self.failures += 1
            self.last_decision_meta = {"is_fallback": True, "fallback_reason": "unparsed_answer",
                                       "answers": ans}
            return self._fallback(ctx)

        gated = False
        anchor_val = self.anchor(ctx)
        if self.cfg.conf_threshold is not None and conf is not None and conf < self.cfg.conf_threshold:
            order = anchor_val
            gated = True
            self.gate_fires += 1
        if self.cfg.anchor_margin is not None:
            a = anchor_val
            order = int(min(max(order, a - self.cfg.anchor_margin), a + self.cfg.anchor_margin))

        order = int(max(0, order))
        self.last_decision_meta = {
            "is_fallback": False, "order": order, "confidence": conf,
            "gate_fired": gated, "anchor": anchor_val, "detail": detail, "answers": ans,
            "usage": usage,
        }
        self.traces.append(self.last_decision_meta)
        return order

    def _compose(self, ctx: dict, ans: dict, meta: dict) -> tuple[int | None, float | None, dict]:
        compose = meta.get("compose")
        if compose in ("grid", "mult"):
            key = "order" if compose == "grid" else "coverage"
            a = ans.get(key) or {}
            probs = a.get("probabilities") or {}
            conf = a.get("confidence")
            if not probs:
                return None, conf, {"stage": "no_probabilities"}
            if self.cfg.mode == "expectation":
                val = sum(float(k) * float(p) for k, p in probs.items())
            else:
                val = float(a.get("choice"))
            if compose == "grid":
                return round(val), conf, {"choice": a.get("choice"), "value": val, "probs": probs}
            # multiplier mode: k * forecast - stock position (arithmetic stays in code)
            ip = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
            qhat = self.es_forecast(ctx)
            return int(round(max(0.0, val * qhat - ip))), conf, {
                "choice": a.get("choice"), "kappa": val, "qhat": round(qhat, 2), "ip": ip, "probs": probs}
        if compose == "reads":
            below = (ans.get("below_target") or {}).get("noul")
            press = (ans.get("backlog_pressure") or {}).get("score")
            trend = (ans.get("demand_trend") or {}).get("score")
            if below is None:
                return None, None, {"stage": "no_reads"}
            qhat = self.es_forecast(ctx)
            lam_eff = max(0.2, self.cfg.lam)
            mult = 1.0
            if trend is not None:
                mult += 0.25 * (float(trend) - 1.0)      # falling -0.25 / flat 0 / rising +0.5
            if below > 0.5:
                mult += 0.25
            if press is not None:
                mult += 0.25 * float(press)              # 0 / 0.25 / 0.5
            target = mult * self.cfg.theta * qhat / (self.cfg.theta * lam_eff) if qhat > 0 else 0.0
            ip = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
            order = int(round(max(0.0, target - ip)))
            return order, None, {"below_target": below, "backlog_pressure": press,
                                 "demand_trend": trend, "mult": mult, "target": round(target, 2)}
        return None, None, {"stage": "unknown_mode"}
