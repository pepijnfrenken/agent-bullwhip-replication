#!/usr/bin/env python3
"""probe_permutation_invariance.py — the post-training permutation gate
(WAVE7-AUDIT-1 amendment 1).

The trained readout must answer by ORDER VALUE, not by letter position: for
every state, the argmax VALUE must be identical under several permutations of
the option->value grid. Pre-registered gate: >= 90% of states invariant under
>= 3 permutations (plus the canonical ascending render).

This quantifies the spec §5 "option-shuffle control arm" on engine states; the
pre-training version of this probe is what falsified the zero-shot premise
(stock Qwen3-0.6B follows the FORMAT but its argmax is a letter prior).

States: JSONL rows carrying an engine ctx ("ctx": {...}) — e.g. the round-2
on-policy states results/wave7_labels/policy_states_seed0.jsonl (or a labels
slice). Sampled deterministically (stride) down to --n states.

Run (GPU, cluster venv):
  python3 probe_permutation_invariance.py \
      --cases results/wave7_labels/policy_states_seed0.jsonl \
      --adapter results/wave7_sft_seed0/checkpoint-119 \
      --n 512 --perms 3 --gate 0.90 \
      --out results/eval/w7-sft-seed0-perm.json

Exit code 1 when the gate fails (use --gate 0 for a report-only control run).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from agent_bullwhip.decision_readout import (DEFAULT_GRID_ID, PINNED_REVISION, ReadoutModel,
                                             get_grid, render_prompt)

GATE = 0.90


def load_states(path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if isinstance(d.get("ctx"), dict):
                rows.append(d["ctx"])
    if not rows:
        raise SystemExit(f"no rows with a 'ctx' dict in {path}")
    stride = max(1, len(rows) // n)
    return rows[::stride][:n]


def build_grids(perms: int, seed: int, grid_id: str = DEFAULT_GRID_ID) -> list[list[int]]:
    """Canonical ascending grid + `perms` seeded shuffled variants (never equal
    to the canonical grid; deduplicated). `grid_id` = the choice set (WAVE8 --grid)."""
    base = list(get_grid(grid_id).values)
    grids = [list(base)]
    rng = random.Random(seed)
    while len(grids) < perms + 1:
        g = list(base)
        rng.shuffle(g)
        if g != list(base) and g not in grids:
            grids.append(g)
    return grids


def main() -> int:
    ap = argparse.ArgumentParser(description="post-training permutation gate (AUDIT-1 fix 1)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--grid", default=DEFAULT_GRID_ID, help="choice set (default wave-7 w7_u21)")
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--cases", required=True,
                    help="JSONL with rows carrying {'ctx': <engine observation>}")
    ap.add_argument("--n", type=int, default=512, help="states sampled (stride)")
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gate", type=float, default=GATE,
                    help="value-invariance gate (0 = report only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seed", type=int, default=20261006, help="grid-shuffle seed")
    args = ap.parse_args()

    states = load_states(Path(args.cases), args.n)
    grids = build_grids(args.perms, args.seed, args.grid)
    rm = ReadoutModel(args.model, device=args.device, dtype=args.dtype,
                      adapter_dir=args.adapter, revision=args.revision or None,
                      grid_id=args.grid)
    # Trained adapters: mass recorded, not asserted (WAVE7_SPEC A1). Untrained:
    # the D2 >=0.99 format proof stays fatal.
    require_mass = 0.0 if args.adapter else 0.99

    values: list[list[int]] = []      # per grid: argmax VALUE per state
    masses: list[float] = []
    for k, g in enumerate(grids):
        prompts = [render_prompt(ctx, rm.tokenizer, grid=g, grid_id=args.grid) for ctx in states]
        vals: list[int] = []
        for i in range(0, len(prompts), args.batch):
            for rec in rm.readout_batch(prompts[i:i + args.batch],
                                        require_mass=require_mass):
                vals.append(int(g[rec["argmax_index"]]))
                masses.append(float(rec["mass"]))
        values.append(vals)
        print(f"grid {k}/{len(grids) - 1}: done ({len(vals)} states)")

    n = len(states)
    invariant = sum(1 for s in range(n)
                    if len({values[k][s] for k in range(len(grids))}) == 1)
    rate = invariant / n if n else 0.0
    disagree = [sum(1 for s in range(n) if values[k][s] != values[0][s])
                for k in range(len(grids))]
    ok = (rate >= args.gate) if args.gate > 0 else True
    arm = args.adapter or "untrained"

    print(f"permutation gate: states={n} grids={len(grids)} invariant={invariant} "
          f"rate={rate:.3f} gate={args.gate:.2f} {'PASS' if ok else 'FAIL'} (arm={arm})")
    print("disagreement vs canonical, per shuffled grid:", disagree[1:])
    print(f"min_mass={min(masses):.4f} (require_mass={require_mass})")

    summary = {
        "arm": arm, "adapter": args.adapter, "cases": str(args.cases),
        "n_states": n, "n_grids": len(grids), "perms": len(grids) - 1,
        "grid_seed": args.seed, "grids": grids,
        "invariant": invariant, "invariance_rate": rate,
        "gate": args.gate, "pass": bool(ok),
        "disagree_vs_canonical": disagree,
        "min_mass": min(masses), "require_mass": require_mass,
        "values": values,
    }
    if args.grid != DEFAULT_GRID_ID:
        summary["grid_id"] = args.grid
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
        print(f"wrote {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
