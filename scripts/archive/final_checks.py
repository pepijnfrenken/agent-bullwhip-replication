#!/usr/bin/env python3
"""Final leak-free checks: paired stats, gate firing, fallback rates, baseline delta."""
import json, glob
from collections import defaultdict, Counter
import itertools, math

files = glob.glob("results/noisy_leakfree/*.jsonl")
path = files[0]
lines = [json.loads(l) for l in open(path) if l.strip()]
comp = [r for r in lines if r.get("total_cost") is not None]
per = defaultdict(list)
for r in comp:
    per[r["config"]].append(r["total_cost"])

def sign_test(a, b):
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    wins = sum(1 for d in diffs if d < 0)
    # two-sided binomial
    p = 2 * sum(math.comb(n, k) * 0.5**n for k in range(min(wins, n - wins) + 1))
    return min(p, 1.0), wins, n

print("=== Paired stats: LLM config vs order_up_to (leak-free noisy) ===")
floor = per["order_up_to"]
for c in ["kb_system_gated", "kb_pointer_verbal"]:
    costs = per[c]
    p, wins, n = sign_test(costs, floor)
    mdiff = sum(x - y for x, y in zip(costs, floor)) / n
    print(f"  {c}: mean_delta={mdiff:+.0f} ({mdiff/ (sum(floor)/len(floor))*100:+.1f}%), wins={wins}/{n}, sign p={p:.4f}")

print("\n=== Baseline: leak-tainted (5.63e11?) vs leak-free (3.02e9) ===")
base = per["baseline"]
bm = sum(base)/len(base)
print(f"  leak-free baseline mean: {bm:,.0f} | median: {sorted(base)[len(base)//2]:,.0f}")

print("\n=== Verbal gate + fallback analysis (leak-free traces) ===")
gated_verbal = gated_conf = fallback = 0
total_decisions = 0
for r in comp:
    if r["config"] != "kb_pointer_verbal":
        continue
    for role, trs in (r.get("traces") or {}).items():
        for t in trs:
            total_decisions += 1
            if t.get("gated_verbal"):
                gated_verbal += 1
            if t.get("gated") and not t.get("gated_verbal"):
                gated_conf += 1
            # failure fallback = order_used came from _fallback (mirror) — detect via failures
print(f"  total verbal decisions: {total_decisions}")
print(f"  gated_verbal fires: {gated_verbal} | conf gate fires: {gated_conf}")

# fallback: count failures per run (failures field in record)
fail_counts = [r.get("failures", {}) for r in comp if r["config"] == "kb_pointer_verbal"]
tot_fail = sum(sum(v.values()) if isinstance(v, dict) else v for v in fail_counts)
print(f"  total format failures (mirror fallbacks): {tot_fail} of {total_decisions} ({tot_fail/max(total_decisions,1)*100:.0f}%)")

print("\n=== Does the verbal gate explain the floor-tying? (order-0-with-backlog events) ===")
o0b = 0
for r in comp:
    if r["config"] != "kb_pointer_verbal":
        continue
    for role, trs in (r.get("traces") or {}).items():
        for t in trs:
            if t.get("order_used") == 0 and (t.get("ctx") or {}).get("backlog", 0) > 0:
                o0b += 1
print(f"  order-0-with-backlog after gates: {o0b} (should be near 0 if gate catches all)")
