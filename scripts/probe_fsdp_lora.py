#!/usr/bin/env python3
"""probe_fsdp_lora.py — two-step FSDP acceptance probe (docs/waves/W8B-FSDP-SCOPE.md).

The point is not that the process exits. A port that runs and trains nothing is
the silent failure this project already has on record, so this asserts gradient
flow explicitly:

  P1 both steps complete and both losses are finite
  P2 LoRA lora_A/lora_B gradients after step 1 are non-zero and finite
  P3 step 2's loss differs from step 1's (the optimizer step had an effect)
  P4 wall time per step under 90 s

Run under torchrun with W8B_FSDP=1, one process per GPU:
  W8B_FSDP=1 torchrun --nproc_per_node=2 --standalone scripts/probe_fsdp_lora.py

Exit 0 only if every criterion passes; prints PROBE PASS or PROBE FAIL.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "Qwen/Qwen3-8B"
REV = "b968826d9c46dd6066d109eabc6255188de91218"
LABELS = ROOT / "results/wave8/S_nom.jsonl"
GRID = "w8_g26b"


def load_rows(n: int) -> list[dict]:
    rows = []
    with open(LABELS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"ctx": r["ctx"], "episode": r.get("episode", 0),
                         "week": r.get("week", 0), "role": r.get("role", "retailer")})
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} label rows available, need {n}")
    return rows


def main() -> int:
    from agent_bullwhip.decision_train import _Policy

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    if world < 2:
        print("FATAL: probe needs at least 2 ranks", flush=True)
        return 2

    pol = _Policy(MODEL, revision=REV, phase="sft", init="fresh",
                  micro_batch=1, grad_ckpt=False, grid_id=GRID)
    n_opt = len(pol.grid)
    rows = load_rows(2)
    targets = [[1.0 / n_opt] * n_opt for _ in rows]   # uniform: any finite target
    # gives a real gradient through the label block

    losses, grads = [], []
    for step in (0, 1):
        t0 = time.time()
        res = pol.sft_step(rows, targets, 1e-5, 991, 0, step)
        dt = time.time() - t0
        losses.append(res.loss)
        finite = all(isinstance(x, float) and x == x and abs(x) != float("inf")
                     for x in (res.loss, res.grad_norm))
        if rank == 0:
            print(f"[probe] step {step}: loss={res.loss:.6f} "
                  f"grad_norm={res.grad_norm:.6f} wall={dt:.1f}s finite={finite}",
                  flush=True)
        if not finite:
            print("PROBE FAIL: non-finite loss or grad_norm", flush=True)
            return 1
        if dt > 90.0:
            print(f"PROBE FAIL: step took {dt:.1f}s, over the 90 s bar", flush=True)
            return 1
        if step == 0:
            nz = 0
            tot = 0
            for name, p in pol.model.named_parameters():
                if "lora_" not in name or p.grad is None:
                    continue
                tot += 1
                g = p.grad.float()
                if bool((g != 0).any()) and bool(pol._torch.isfinite(g).all()):
                    nz += 1
            grads.append((nz, tot))
            if rank == 0:
                print(f"[probe] lora params with non-zero finite grad: {nz}/{tot}",
                      flush=True)
            if tot == 0 or nz == 0:
                print("PROBE FAIL: LoRA gradients absent or all zero — the port "
                      "runs but trains nothing", flush=True)
                return 1

    if losses[0] == losses[1]:
        print("PROBE FAIL: identical loss across the two steps", flush=True)
        return 1

    if rank == 0:
        print(f"PROBE PASS: 2 steps, losses {losses[0]:.6f} -> {losses[1]:.6f}, "
              f"lora grads {grads[0][0]}/{grads[0][1]} live", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
