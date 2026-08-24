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
    "baseline": {},  # default decentralized LLM
    "voting5": {"voting": 5},
    "voting10": {"voting": 10},
    "guardrail": {"guardrail_ratio": 2.0},
    "anchor": {"anchor_margin": 6},
    "prompt_weighted": {"prompt_variant": "weighted"},
    "combined": {"guardrail_ratio": 2.0, "anchor_margin": 6, "prompt_variant": "weighted"},
    "mirror": {"_baseline": "mirror"},
    "order_up_to": {"_baseline": "order_up_to"},
}


def make_agents(config: dict, model: str) -> dict:
    if config.get("_baseline") == "mirror":
        return {r: MirrorAgent() for r in ROLES}
    if config.get("_baseline") == "order_up_to":
        return {r: OrderUpToAgent() for r in ROLES}
    lcfg = {k: v for k, v in config.items() if not k.startswith("_")}
    return {r: LLMAgent(r, LLMAgentConfig(**lcfg, model=model)) for r in ROLES}


def run_config(name: str, runs: int, model: str, horizon: int, pattern: str, outdir: Path) -> dict:
    cfg = CONFIGS[name]
    demand = make_demand(horizon, pattern)
    sim_cfg = SimConfig(horizon=horizon)
    results: list[dict] = []
    run_logs = []
    t0 = time.time()
    for i in range(runs):
        agents = make_agents(cfg, model)
        log = run_game(agents, demand, sim_cfg)
        log.agents = agents  # attach for failure metrics
        run_logs.append(log)
        results.append({
            "run": i,
            "config": name,
            "model": model,
            "total_cost": log.total_cost(),
            "orders": {r: log.orders(r) for r in ROLES},
            "backlogs": {r: log.backlogs(r) for r in ROLES},
            "failures": {r: agents[r].failures for r in ROLES},
            "calls": {r: agents[r].calls for r in ROLES},
        })
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

    metrics = compute_metrics(run_logs)
    metrics.update({"config": name, "model": model, "runs": runs, "wall_sec": time.time() - t0})
    summary_path = outdir / f"{name}.summary.json"
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
