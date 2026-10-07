#!/usr/bin/env python3
"""Deep trace collector for the ToolAgent — captures EVERYTHING.

Runs N full games of the tool agent on the step pattern through the REAL engine
(run_game) with a recording wrapper, and dumps per decision:
  - role, week, full ctx (exactly what the model saw)
  - the Python code the model wrote (full)
  - the exec result (stdout or error)
  - the final order it returned
  - what the deterministic floor (order_up_to 3.0, 0.5) ordered for the SAME state
  - the delta (model vs floor) → mistake label
  - whether a tool error occurred, and its type

Output: results/toolagent_traces/<run>.jsonl — one JSON per decision.
"""
import json, os, time
from pathlib import Path
from collections import Counter

from agent_bullwhip.agents import ToolAgent, LLMAgentConfig, OrderUpToAgent, build_prompt
from agent_bullwhip.engine import make_demand, SimConfig, run_game, ROLES

OUT = Path("results/toolagent_traces")
N_RUNS = 1
HORIZON = 36
PATTERN = "step"
MODEL = "Qwen/Qwen3.8-Flash"


def classify_error(result: str) -> str:
    r = result.lower()
    for kind in ["nameerror", "typeerror", "valueerror", "indexerror",
                 "keyerror", "zerodivisionerror", "attributeerror",
                 "syntaxerror", "indentationerror", "import error",
                 "unboundlocalerror", "recursionerror", "overflowerror"]:
        if kind in r:
            return kind
    if "error" in r:
        return "other_error"
    return "none"


class RecordingToolAgent(ToolAgent):
    """ToolAgent that records every decision's ctx + floor comparison."""

    def __init__(self, role, cfg, floor, checkpoint_file=None):
        super().__init__(role, cfg)
        self.floor = floor
        self.records = []
        self.ctx_log = []
        self._checkpoint_file = checkpoint_file

    def decide(self, ctx):
        t0 = time.time()
        q = super().decide(ctx)
        dt = time.time() - t0
        # floor's order for the same state (its own running forecast)
        try:
            floor_order = self.floor(ctx)
        except Exception:
            floor_order = None
        tool_trace = self.tool_traces[-1] if self.tool_traces else []
        code_written = ""
        exec_result = ""
        tool_err = "none"
        reasoning = ""
        for tc in tool_trace:
            args = tc.get("arguments", {})
            code_written = args.get("code", "") if isinstance(args, dict) else str(args)
            exec_result = tc.get("result", "")
            reasoning = tc.get("reasoning", "") or reasoning
            if exec_result.startswith("ERROR"):
                tool_err = classify_error(exec_result)
        # derive WHY-context features from ctx (for mistake correlation)
        oh, b, o = ctx.get("on_hand", 0), ctx.get("backlog", 0), ctx.get("outstanding", 0)
        net_pos = oh - b + o
        last = ctx.get("last_orders") or []
        feats = {
            "net_pos": net_pos,
            "low_inventory": oh - b <= 0,          # service failure risk
            "has_backlog": b > 0,
            "last_orders_recent_mean": (sum(last[-3:]) / len(last[-3:])) if last else None,
            "incoming_now": ctx.get("incoming_now"),
            "week": ctx.get("t"),
        }
        rec = {
            "role": self.role, "t": ctx.get("t"),
            "ctx": ctx,
            "features": feats,
            "floor_order": floor_order,
            "model_order": q,
            "delta": (q - floor_order) if q is not None and floor_order is not None else None,
            "tool_error": tool_err,
            "code": code_written,
            "exec_result": exec_result,
            "latency_s": round(dt, 2),
            "tool_calls": tool_trace,
            "reasoning": reasoning,
            "decision_meta": self.last_decision_meta,   # is_fallback / mirror_after_error
            "prompt": build_prompt(self.role, ctx, self.cfg.prompt_variant),
        }
        self.records.append(rec)
        self.ctx_log.append(ctx)
        # checkpoint: write this decision immediately so a mid-game outage
        # doesn't lose the whole run (resume = re-open the file in append mode).
        if self._checkpoint_file:
            with open(self._checkpoint_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    decision_count = 0
    model = os.environ.get("TOOLAGENT_MODEL", MODEL)
    n_runs = int(os.environ.get("TOOLAGENT_RUNS", str(N_RUNS)))
    for run in range(n_runs):
        floor = OrderUpToAgent(theta=3.0, lam=0.5)
        out_file = OUT / f"run{run}.jsonl"
        # resume-friendly: truncate only if this is a fresh run; if the file
        # already has decisions for this run, append (checkpointing resumes).
        agents = {r: RecordingToolAgent(
            r, LLMAgentConfig(model=model, tag="trace"), floor,
            checkpoint_file=out_file) for r in ROLES}
        demand = make_demand(HORIZON, PATTERN)
        log = run_game(agents, demand, SimConfig(horizon=HORIZON))
        # summary (the per-decision JSONL was already checkpointed above)
        with open(OUT / f"run{run}.summary.json", "w") as f:
            json.dump({"n_runs": 1, "mean_cost": log.total_cost(),
                       "model": model, "pattern": PATTERN}, f)
        # stats
        for r in ROLES:
            for rec in agents[r].records:
                decision_count += 1
                if rec["model_order"] is None:
                    stats["model_parse_fail"] += 1
                elif rec["floor_order"] is not None and rec["model_order"] == rec["floor_order"]:
                    stats["match_floor"] += 1
                if rec["tool_error"] != "none":
                    stats[f"tool_err_{rec['tool_error']}"] += 1
                if rec["tool_calls"]:
                    stats["used_tool"] += 1
                else:
                    stats["no_tool_call"] += 1
        print(f"run {run} done — cost {log.total_cost():,.0f}", flush=True)

    print("\n=== STATS ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if decision_count:
        print(f"\n  match_floor rate: {stats['match_floor']/decision_count:.1%}")
        print(f"  tool use rate: {stats['used_tool']/decision_count:.1%}")


if __name__ == "__main__":
    main()
