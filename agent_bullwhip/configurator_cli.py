"""CLI for the LLM-as-configurator experiment (Wave 3 idea #1).

Usage:
    python -m agent_bullwhip.configurator_cli --runs 30 --model deepseek-v4-flash
    python -m agent_bullwhip.configurator_cli --runs 30 --model qwen3.6-35b --blind
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .client import list_models
from .configurator import run_configurator_game
from .engine import make_demand
from .metrics import compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--model", default=os.environ.get("FREEINFERENCE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--pattern", default="step")
    ap.add_argument("--blind", action="store_true", help="no demand context given to configurator")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", action="store_true", help="list endpoint models and exit")
    args = ap.parse_args()

    if args.models:
        for m in list_models():
            print(m)
        return

    demand = make_demand(args.horizon, args.pattern)
    demand_summary = (f"demand is {args.pattern} pattern over {args.horizon} weeks: "
                      f"first 4 weeks at 4, then 8 (step); total mean ~{sum(demand)/len(demand):.1f}")
    if args.blind:
        demand_summary = ""

    tag = f"{args.model}-configurator{'blind' if args.blind else ''}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = []
    total_pt = total_ct = 0
    for i in range(args.runs):
        res = run_configurator_game(args.model, demand, args.horizon, blind=args.blind,
                                    demand_summary=demand_summary)
        total_pt += res["prompt_tokens"]
        total_ct += res["completion_tokens"]
        runs.append(res["log"])
        with open(outdir / f"{tag}.jsonl", "a") as f:
            f.write(json.dumps({
                "run": i, "config": "configurator", "model": args.model, "tag": tag,
                "total_cost": res["total_cost"], "params": res["params"],
                "failures": res["failures"], "tokens": {"prompt": res["prompt_tokens"],
                                                        "completion": res["completion_tokens"]},
            }) + "\n")

    metrics = compute_metrics(runs)
    metrics.update({
        "config": "configurator", "model": args.model, "tag": tag, "runs": args.runs,
        "completed": len(runs), "failed": args.runs - len(runs),
        "tokens": {"prompt": total_pt, "completion": total_ct, "total": total_pt + total_ct},
    })
    (outdir / f"{tag}.summary.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2, default=float))


if __name__ == "__main__":
    main()
