#!/usr/bin/env python3
"""Gate effectiveness + KB-principle invocation in traces."""
import json, statistics, collections, re

def load(cfg):
    runs = []
    for line in open(f"results/deepseek-v4-flash-{cfg}.jsonl"):
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    return runs

def order_up_to_order(ctx):
    last = ctx.get("last_orders") or []
    if not last:
        return 0
    smoothed = last[0]
    for v in last[1:]:
        smoothed = 0.5 * smoothed + 0.5 * v
    target = 3.0 * smoothed + ctx.get("backlog", 0)
    position = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
    return max(0, int(round(target - position)))

print("=" * 80)
print("A. Does the gate actually prevent catastrophic under-orders?")
print("   (non-gated vs gated configs, order_used=0 with backlog>0)")
print("=" * 80)
for cfg in ["kb_introspect", "kb_system_introspect", "kb_pointer_introspect",
            "kb_introspect_gated", "kb_system_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    n_total = n_bad = 0
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                n_total += 1
                if t.get("order_used") == 0 and (t.get("ctx") or {}).get("backlog", 0) > 0:
                    n_bad += 1
    print(f"{cfg:<24} order0-with-backlog: {n_bad:>4}/{n_total:>5} ({100*n_bad/n_total:4.1f}%)")

print()
print("=" * 80)
print("B. KB-principle invocation in thinking/reasoning text")
print("=" * 80)
KB_MARKERS = {
    "kb-1 (don't over-react)": r"over-?react",
    "kb-2 (cover backlog, cap)": r"cap|2x|backlog first",
    "kb-3 (account pipeline)": r"pipelin|outstanding|in-transit|in flight|in-flight",
    "kb-3b (in-transit NOT available)": r"in-transit is not|not available this week|arriv(e|es|ing) (in|next)",
    "kb-4 (low conf fallback)": r"fallback|do no harm|conservative|safe",
    "kb-5 (EMA smoothing)": r"smooth|ema|moving average",
    "order-up-to": r"order-?up-?to",
}
for cfg in ["kb_introspect_gated", "kb_system_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    counts = collections.Counter()
    n_think = 0
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                text = ((t.get("thinking") or "") + " " + (t.get("reasoning") or "")).lower()
                if text.strip():
                    n_think += 1
                for label, pat in KB_MARKERS.items():
                    if re.search(pat, text):
                        counts[label] += 1
    print(f"--- {cfg} (n_traces_with_text={n_think}) ---")
    for label, c in counts.most_common():
        print(f"   {label:<38} {c:>5} ({100*c/max(n_think,1):4.1f}%)")

print()
print("=" * 80)
print("C. Role-level mean |dev| from floor (who is the weakest link?)")
print("=" * 80)
for cfg in ["kb_introspect_gated", "kb_system_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    by_role = collections.defaultdict(list)
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                dev = abs(t.get("order_used") - order_up_to_order(t["ctx"]))
                by_role[role].append(dev)
    print(f"--- {cfg} ---")
    for role in ["retailer", "wholesaler", "distributor", "factory"]:
        devs = by_role[role]
        if devs:
            print(f"   {role:<12} mean|dev|={statistics.mean(devs):6.1f}  median={statistics.median(devs):5.0f}")
