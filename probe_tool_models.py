#!/usr/bin/env python3
"""Probe tool-calling quality across candidate models (real API, bounded cost).

For each model in MODELS, run ONE game (4 roles x 36 weeks = 144 decisions) of
the ToolAgent against the step demand, and report:

  - tool_call_rate      : fraction of decisions where the model called the tool
  - code_error_rate     : fraction of tool calls whose exec returned an ERROR
  - retry_after_error   : fraction of errored decisions where the model made a
                          SECOND tool call (the recovery loop actually engaged)
  - parse_fail_rate     : fraction of decisions with no parseable integer
  - match_floor_rate    : fraction matching the deterministic order-up-to floor
  - mean_cost / cv      : game cost vs 3,681 floor

Usage:
    python3 probe_tool_models.py [--quick] [--models glm-5.3-flash,minimax-m3]
"""
from __future__ import annotations

import argparse, json, os, time
from collections import Counter
from pathlib import Path

# ensure the API key is available to the client even when not in env
_env_local = Path(__file__).resolve().parent / ".env.local"
if _env_local.exists():
    for _line in _env_local.read_text().splitlines():
        if _line.startswith("FREEINFERENCE_API_KEY=") and not os.environ.get("FREEINFERENCE_API_KEY"):
            os.environ["FREEINFERENCE_API_KEY"] = _line.split("=", 1)[1].strip()

from agent_bullwhip.agents import ToolAgent, LLMAgentConfig, OrderUpToAgent
from agent_bullwhip.engine import ROLES, SimConfig, make_demand, run_game

OUT = Path("results/toolagent_probe")
HORIZON = 36
PATTERN = "step"
DEFAULT_MODELS = ["glm-5.3-flash", "minimax-m3", "qwen3.6-35b", "deepseek-v4-flash"]

class ProbeAgent(ToolAgent):
    """ToolAgent that also records per-decision tool stats (no floor needed)."""
    def __init__(self, role, cfg):
        super().__init__(role, cfg)
        self.decision_stats = []

    def decide(self, ctx):
        t0 = time.time()
        q = super().decide(ctx)
        trace = self.tool_traces[-1] if self.tool_traces else []
        calls = [t for t in trace if isinstance(t, dict)]
        n_calls = len(calls)
        n_err = sum(1 for t in calls if str(t.get("result", "")).startswith("ERROR"))
        # retry = a second tool call happened after an error
        retry = False
        seen_err = False
        for t in calls:
            if str(t.get("result", "")).startswith("ERROR"):
                seen_err = True
            elif seen_err:
                retry = True
        self.decision_stats.append({
            "t": ctx.get("t"), "q": q,
            "n_calls": n_calls, "n_err": n_err, "retry_after_error": retry,
            "is_fallback": self.last_decision_meta.get("is_fallback", False),
            "mirror_after_error": self.last_decision_meta.get("mirror_after_error", False),
            "latency_s": round(time.time() - t0, 2),
        })
        return q

def run_one(model: str, quick: bool) -> dict:
    horizon = 12 if quick else HORIZON  # quick = 1 role * 12 weeks smoke test
    roles = ["retailer"] if quick else ROLES
    demand = make_demand(horizon, PATTERN)
    floor = OrderUpToAgent(theta=3.0, lam=0.5)
    agents = {r: ProbeAgent(r, LLMAgentConfig(model=model, tag="probe")) for r in roles}
    sim = SimConfig(horizon=horizon)
    log = run_game(agents, demand, sim)
    stats = []
    for r in roles:
        stats += agents[r].decision_stats
    n = len(stats)
    cost = log.total_cost()
    # floor cost for same horizon/pattern (fresh floor per role)
    floor_cost = None
    try:
        fl = {r: OrderUpToAgent(theta=3.0, lam=0.5) for r in roles}
        flog = run_game(fl, demand, SimConfig(horizon=horizon))
        floor_cost = flog.total_cost()
    except Exception:
        pass
    n_calls = sum(s["n_calls"] for s in stats)
    n_err = sum(s["n_err"] for s in stats)
    n_retry = sum(1 for s in stats if s["retry_after_error"])
    n_fb = sum(1 for s in stats if s["is_fallback"])
    n_mirror = sum(1 for s in stats if s["mirror_after_error"])
    match = sum(1 for s in stats if s["q"] is not None and floor is not None)
    # match vs floor needs per-decision floor; approximate via agent comparison is
    # skipped here — we report the cost ratio instead, which is the game metric.
    return {
        "model": model, "quick": quick, "decisions": n,
        "tool_call_rate": round(n_calls / n, 3) if n else None,
        "avg_calls_per_decision": round(n_calls / n, 2) if n else None,
        "code_error_rate": round(n_err / n_calls, 3) if n_calls else None,
        "retry_after_error_rate": round(n_retry / max(1, n_err), 3) if n_err else None,
        "parse_fail_rate": round(n_fb / n, 3) if n else None,
        "mirror_after_error_rate": round(n_mirror / n, 3) if n else None,
        "cost": cost, "floor_cost": floor_cost,
        "cost_ratio_vs_floor": round(cost / floor_cost, 3) if floor_cost else None,
        "wall_s": None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--quick", action="store_true", help="12-week single-role smoke test")
    ap.add_argument("--out", default=str(OUT / "probe_results.json"))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for m in models:
        print(f"\n=== probing {m} (quick={args.quick}) ===", flush=True)
        t0 = time.time()
        try:
            r = run_one(m, args.quick)
            r["wall_s"] = round(time.time() - t0, 1)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "quick"}, indent=2))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({"model": m, "error": str(e)})
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    # ranking
    ok = [r for r in results if "cost_ratio_vs_floor" in r]
    if ok:
        ok.sort(key=lambda r: (r.get("code_error_rate") or 1, r.get("cost_ratio_vs_floor") or 99))
        print("\n=== RANKING (by code-error rate, then cost ratio) ===")
        for r in ok:
            print(f"  {r['model']:<22} err={r.get('code_error_rate')}  "
                  f"retry={r.get('retry_after_error_rate')}  "
                  f"cost_ratio={r.get('cost_ratio_vs_floor')}  "
                  f"tool_call={r.get('tool_call_rate')}")

if __name__ == "__main__":
    main()
