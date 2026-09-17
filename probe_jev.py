#!/usr/bin/env python3
"""probe_jev.py — smoke test + state-level probe for the TypeSafe Jev agent.

Runs the 6 representative beer-game states (reused from run_model_comparison)
through each Jev mode and prints, per state: the model's chosen option, the full
probability distribution, the confidence, the composed order, and the anchor for
comparison. Appends one JSON line per (state, mode) to results/jev_probe/.

Usage:
    python3 probe_jev.py                       # all modes, all states
    python3 probe_jev.py --modes choice_mult   # one mode
    python3 probe_jev.py --grid-max 60 --states 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent_bullwhip.jev_agent import JevAgent, JevAgentConfig, ask, _load_key
from run_model_comparison import STATES  # module-level constants only; no network at import

OUT = Path("results/jev_probe")
OUT.mkdir(parents=True, exist_ok=True)

MODES = ("choice_mult", "choice_grid", "expectation", "reads")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--states", type=int, default=len(STATES))
    ap.add_argument("--grid-max", type=int, default=40)
    ap.add_argument("--grid-step", type=int, default=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the request payload only")
    args = ap.parse_args()

    if not _load_key() and not args.dry_run:
        raise SystemExit(
            "TYPESAFE_API_KEY missing. Add it to the environment or to "
            ".env.local (TYPESAFE_API_KEY=...) — get it from console.typesafe.ai.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    states = STATES[: args.states]
    for mode in modes:
        tag = f"jev-{mode}"
        fpath = OUT / f"{tag}.jsonl"
        for i, ctx in enumerate(states):
            cfg = JevAgentConfig(mode=mode, grid_max=args.grid_max, grid_step=args.grid_step,
                                 model=args.model, tag=tag)
            agent = JevAgent("retailer", cfg)
            if args.dry_run:
                q, _ = agent._questions()  # noqa: SLF001 — dry-run debugging
                print(json.dumps({"mode": mode, "state": agent._state(ctx),  # noqa: SLF001
                                  "questions": q}, indent=1)[:1400])
                continue
            t0 = time.time()
            try:
                order = agent.decide(ctx)
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}] state {i}: FAILED {e}")
                raise
            rec = {"mode": mode, "state_index": i, "ctx": ctx, "order": order,
                   "anchor": agent.anchor(dict(ctx)), "seconds": round(time.time() - t0, 2),
                   "meta": agent.last_decision_meta,
                   "tokens": {"in": agent.prompt_tokens, "out": agent.completion_tokens}}
            with fpath.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            m = agent.last_decision_meta
            det = m.get("detail") or {}
            print(f"[{tag}] state {i} t={ctx.get('t')}: order={order:>4}  anchor={rec['anchor']:>4}  "
                  f"choice={det.get('choice') or det.get('kappa') or '-'}  "
                  f"conf={m.get('confidence')}  {rec['seconds']}s")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
