"""Run N replications of a config and dump JSONL + summary.

Usage:
    python -m agent_bullwhip.runner --config baseline --runs 30 --model deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .agents import LLMAgent, LLMAgentConfig, MirrorAgent, OrderUpToAgent
from .client import list_models
from .engine import ROLES, SimConfig, make_demand, run_game
from .metrics import compute_metrics

CONFIGS: dict[str, dict] = {
    # ---- the paper's own levers (Wave 1) ----
    "baseline": {},  # default decentralized LLM (replication anchor)
    "budget": {"guardrail_ratio": 2.0},  # paper Table 1 "budget" policy (order cap)
    "prompt_weighted": {"prompt_variant": "weighted"},  # paper §3.4 reframe
    "voting5": {"voting": 5},  # paper §4.3 repeated sampling (negative result check)
    # ---- our extension (Wave 2) ----
    "anchor": {"anchor_margin": 6},
    "combined": {"guardrail_ratio": 2.0, "anchor_margin": 6, "prompt_variant": "weighted"},
    # ---- Wave 2: knowledge base + introspection (the iteration loop) ----
    "kb": {"kb": True},                                    # playbook injected, no trace
    "introspect": {"introspect": True},                    # reason + confidence, logged
    "kb_introspect": {"kb": True, "introspect": True},     # both (the full upgrade)
    "kb_introspect_gated": {"kb": True, "introspect": True, "conf_threshold": 0.5,
                            "anchor_margin": 6},           # self-gate: low conf -> anchor
    # ---- Wave 2b: KB placement A/B (system message vs inline user message) ----
    "kb_system": {"kb": True, "kb_placement": "system"},                 # KB in system msg
    "kb_system_introspect": {"kb": True, "kb_placement": "system",
                             "introspect": True},                        # system + trace
    "kb_system_gated": {"kb": True, "kb_placement": "system",
                        "introspect": True, "conf_threshold": 0.5,
                        "anchor_margin": 6},              # system + self-gate
    # ---- Wave 2c: pointer-only KB (no contents injected — self-directed) ----
    "kb_pointer": {"kb": True, "kb_placement": "pointer"},               # guide-only, no contents
    "kb_pointer_introspect": {"kb": True, "kb_placement": "pointer",
                              "introspect": True},                       # pointer + trace
    "kb_pointer_gated": {"kb": True, "kb_placement": "pointer",
                         "introspect": True, "conf_threshold": 0.5,
                         "anchor_margin": 6},            # pointer + self-gate
    # ---- deterministic baselines ----
    "mirror": {"_baseline": "mirror"},
    "order_up_to": {"_baseline": "order_up_to"},
}


def make_agents(config: dict, model: str, tag: str | None = None) -> dict:
    if config.get("_baseline") == "mirror":
        return {r: MirrorAgent() for r in ROLES}
    if config.get("_baseline") == "order_up_to":
        return {r: OrderUpToAgent() for r in ROLES}
    lcfg = {k: v for k, v in config.items() if not k.startswith("_")}
    lcfg["tag"] = tag or model
    return {r: LLMAgent(r, LLMAgentConfig(**lcfg, model=model)) for r in ROLES}


def run_config(name: str, runs: int, model: str, horizon: int, pattern: str, outdir: Path) -> dict:
    cfg = CONFIGS[name]
    tag = f"{model}-{name}"
    demand = make_demand(horizon, pattern)
    sim_cfg = SimConfig(horizon=horizon)
    results: list[dict] = []
    run_logs = []
    t0 = time.time()
    total_prompt_tokens = total_completion_tokens = 0
    for i in range(runs):
        try:
            agents = make_agents(cfg, model, tag=tag)
            log = run_game(agents, demand, sim_cfg)
            log.agents = agents  # attach for failure metrics
            run_logs.append(log)
            prompt_tok = sum(getattr(a, "prompt_tokens", 0) for a in agents.values())
            comp_tok = sum(getattr(a, "completion_tokens", 0) for a in agents.values())
            total_prompt_tokens += prompt_tok
            total_completion_tokens += comp_tok
            results.append({
                "run": i,
                "config": name,
                "model": model,
                "tag": tag,
                "total_cost": log.total_cost(),
                "orders": {r: log.orders(r) for r in ROLES},
                "backlogs": {r: log.backlogs(r) for r in ROLES},
                "failures": {r: agents[r].failures for r in ROLES},
                "calls": {r: agents[r].calls for r in ROLES},
                "tokens": {"prompt": prompt_tok, "completion": comp_tok},
                # Wave 2: per-decision traces (order/confidence/reasoning/gated) for gap mining
                "traces": {r: getattr(agents[r], "traces", []) for r in ROLES},
            })
        except Exception as e:  # noqa: BLE001 - a failed run must not kill the matrix
            results.append({"run": i, "config": name, "model": model, "tag": tag,
                            "total_cost": None, "error": str(e)})
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / f"{tag}.jsonl", "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

    try:
        if run_logs:
            metrics = compute_metrics(run_logs)
        else:
            metrics = {"n_runs": 0, "mean_cost": None, "cv_cost": None, "error": "all runs failed"}
    except Exception as e:  # noqa: BLE001 - metrics failure must not kill the matrix
        metrics = {"n_runs": len(run_logs), "error": f"metrics failed: {e}"}
    metrics.update({
        "config": name, "model": model, "tag": tag, "runs": runs,
        "completed": len(run_logs), "failed": len(results) - len(run_logs),
        "wall_sec": time.time() - t0,
        "tokens": {"prompt": total_prompt_tokens, "completion": total_completion_tokens,
                   "total": total_prompt_tokens + total_completion_tokens},
    })
    summary_path = outdir / f"{tag}.summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="baseline")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("FREEINFERENCE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--pattern", default="step")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", action="store_true", help="list available endpoint models and exit")
    args = ap.parse_args()

    if args.models:
        for m in list_models():
            print(m)
        return

    outdir = Path(args.outdir)
    m = run_config(args.config, args.runs, args.model, args.horizon, args.pattern, outdir)
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=2, default=float))


if __name__ == "__main__":
    main()
