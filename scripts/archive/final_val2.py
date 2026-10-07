#!/usr/bin/env python3
"""Final interleaved val2 results: full table + window-drift measurement."""
import json, glob
from collections import defaultdict

files = glob.glob("results/interleaved_val2/*.jsonl")
if not files:
    print("no jsonl")
    raise SystemExit
path = files[0]
lines = [json.loads(l) for l in open(path) if l.strip()]
comp = [r for r in lines if r.get("total_cost") is not None]
fail = [r for r in lines if r.get("total_cost") is None]
print(f"records: {len(lines)} | completed: {len(comp)} | failed: {len(fail)}")
if fail:
    from collections import Counter
    print("failures by config:", dict(Counter(r["config"] for r in fail)))

# per-config stats
per = defaultdict(list)
for r in comp:
    per[r["config"]].append(r["total_cost"])
print(f"\n{'config':<24}{'n':>3}{'mean':>14}{'std':>12}{'cv':>8}{'min':>12}{'max':>14}")
for c in sorted(per, key=lambda c: sum(per[c])/len(per[c])):
    costs = per[c]
    m = sum(costs)/len(costs)
    sd = (sum((x-m)**2 for x in costs)/len(costs))**0.5
    print(f"{c:<24}{len(costs):>3}{m:>14,.0f}{sd:>12,.0f}{sd/m:>8.3f}{min(costs):>12,.0f}{max(costs):>14,.0f}")

# window drift: cost of the SAME config across passes (the interleaved measure)
print("\n=== window drift: kb_system_gated per pass ===")
for r in sorted([r for r in comp if r["config"] == "kb_system_gated"], key=lambda r: r["run"]):
    print(f"  pass {r['run']}: {r['total_cost']:,.0f}")
print("\n=== window drift: kb_system_introspect per pass ===")
for r in sorted([r for r in comp if r["config"] == "kb_system_introspect"], key=lambda r: r["run"]):
    print(f"  pass {r['run']}: {r['total_cost']:,.0f}")

# gated vs ungated same-pass ratio
print("\n=== gated vs ungated (same pass) ===")
by_pass = defaultdict(dict)
for r in comp:
    by_pass[r["run"]][r["config"]] = r["total_cost"]
for p in sorted(by_pass):
    row = by_pass[p]
    if "kb_system_gated" in row and "kb_system_introspect" in row:
        print(f"  pass {p}: gated={row['kb_system_gated']:,.0f} ungated={row['kb_system_introspect']:,.0f} ratio={row['kb_system_introspect']/row['kb_system_gated']:.1f}x")
