#!/usr/bin/env python3
"""Status check: noisy arm progress, watchers, herdr state."""
import json, os, glob

print("=== noisy arm results so far ===")
for p in sorted(glob.glob("results/*noisy*")):
    try:
        n = sum(1 for l in open(p) if l.strip())
        print(f"  {os.path.basename(p)}: {n} lines")
    except Exception as e:
        print(f"  {os.path.basename(p)}: {e}")

print("\n=== noisy arm summary (if any) ===")
for p in sorted(glob.glob("results/*noisy*.summary.json")):
    try:
        d = json.load(open(p))
        print(f"  {os.path.basename(p)}: completed={d.get('completed')} mean={d.get('mean_cost')} cv={d.get('cv_cost')}")
    except Exception as e:
        print(f"  {os.path.basename(p)}: {e}")

print("\n=== v2 kb_pointer_verbal on disk ===")
n = sum(1 for l in open("results/deepseek-v4-flash-kb_pointer_verbal.jsonl") if l.strip())
print(f"  lines: {n}")
