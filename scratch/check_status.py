#!/usr/bin/env python3
"""Check v2 ablation failures + noisy arm + auditor status."""
import json, os, glob

print("=== v2 kb_pointer_verbal failures ===")
fails = 0
for line in open("results/deepseek-v4-flash-kb_pointer_verbal.jsonl"):
    d = json.loads(line)
    if d.get("error"):
        fails += 1
        print(f"  run {d['run']}: {str(d['error'])[:110]}")
print("total failed runs:", fails)

print("\n=== noisy arm results so far ===")
for p in sorted(glob.glob("results/*noisy*")):
    print(f"  {os.path.basename(p)} ({os.path.getsize(p)} bytes)")

print("\n=== AUDIT.md ===")
print("  exists:", os.path.exists("AUDIT.md"), os.path.getsize("AUDIT.md") if os.path.exists("AUDIT.md") else "")

print("\n=== prompt_hash_dups in v2 ===")
dups = [json.loads(l).get("prompt_hash_dups") for l in open("results/deepseek-v4-flash-kb_pointer_verbal.jsonl")]
print("  total:", sum(d for d in dups if d), "| runs with dups:", sum(1 for d in dups if d))
