#!/usr/bin/env python3
"""Fingerprint a wave-7 label set at several precision levels.

Purpose: tell "byte-identical rows" apart from "identical decisions, ULP-level
float noise" between two machines (hub 5-process run vs Umbrella 96-process run).
Last-bit drift root cause (verified 2026-10-06): CPython 3.12 switched builtin
sum() for floats to Neumaier summation (gh-100425); the labeler's mean/std/
softmax-denominator sums follow the running Python minor (hub 3.13 vs cluster
3.11.3) — not numpy, not glibc, not exp. The manifest `hash` covers full-precision
float text of the first `hash_rows` rows, so a single last-bit difference breaks
it while the decisions and the aggregate stats can still be identical.

Run the SAME command on both machines and compare the printed lines:

    python3 tools/label_fingerprint.py results/wave7_labels/labels.jsonl

Interpretation:
- forward: same on both + `decisions_2000` same + `J_9dp_2000`/`p_9dp_2000` same
  -> benign ULP noise in the last stored digits; the label decisions are one set.
- `decisions_2000` differs -> real disagreement; do not train until understood.
- forward differs, J/p at 9dp differ too -> structural float difference, larger
  than ULP; investigate before training.
"""
import hashlib
import json
import sys
from pathlib import Path


def digest(seq) -> str:
    h = hashlib.sha256()
    for x in seq:
        h.update(json.dumps(x, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def main() -> None:
    path = Path(sys.argv[1])
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    first = rows[:2000]

    decisions = [[r.get("episode"), r.get("week"), r.get("role"),
                  r.get("flat"), r.get("argmin_idx")] for r in first]
    j9 = [[round(float(v), 9) for v in r["J"]] for r in first if "J" in r]
    p9 = [[round(float(v), 9) for v in r["p"]] for r in first if "p" in r]

    print("rows:", len(rows))
    print("decisions_2000:", digest(decisions))
    print("J_9dp_2000:", digest(j9))
    print("p_9dp_2000:", digest(p9))
    print("full_2000:", digest(first))  # self-check: equals the manifest hash

    mpath = path.with_suffix(".manifest.json")
    if mpath.exists():
        m = json.loads(mpath.read_text())
        for k in ("hash", "hash_rows", "n_states", "n_flat", "flat_rate",
                  "edge_argmin_rate", "mean_target_entropy_nats", "processes"):
            if k in m:
                print(f"manifest.{k}: {m[k]!r}")


if __name__ == "__main__":
    main()
