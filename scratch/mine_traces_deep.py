#!/usr/bin/env python3
"""Deep trace mining: compare LLM orders vs deterministic floor policy in same state."""
import json, statistics, collections

CONFIGS = ["kb_introspect_gated", "kb_system_gated", "kb_pointer_gated", "kb_introspect", "kb_general_introspect"]

def load(cfg):
    runs = []
    for line in open(f"results/deepseek-v4-flash-{cfg}.jsonl"):
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    return runs

# Deterministic order-up-to policy (theta=3, lambda=0.5) — from the repo's order_up_to config
def order_up_to_order(ctx):
    # standard beer game order-up-to: target = theta * smoothed demand + backlog
    # smoothed demand via EMA(0.5) over last_orders (incoming)
    last = ctx.get("last_orders") or []
    if not last:
        return 0
    smoothed = last[0]
    for v in last[1:]:
        smoothed = 0.5 * smoothed + 0.5 * v
    theta = 3.0
    target = theta * smoothed + ctx.get("backlog", 0)
    # position = on_hand + outstanding - backlog
    position = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
    return max(0, int(round(target - position)))

print("=" * 80)
print("A. LLM vs ORDER_UP_TO in the same state — deviation distribution")
print("=" * 80)
for cfg in CONFIGS:
    runs = load(cfg)
    devs = []
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                opt = order_up_to_order(t["ctx"])
                dev = t.get("order_used") - opt
                devs.append(dev)
    n = len(devs)
    mean_dev = sum(devs) / n
    over = sum(1 for x in devs if x > 0)
    under = sum(1 for x in devs if x < 0)
    exact = sum(1 for x in devs if x == 0)
    print(f"{cfg:<24} n={n:>5} mean_dev={mean_dev:+7.1f}  over={over:>5} ({100*over/n:4.1f}%)  under={under:>5} ({100*under/n:4.1f}%)  exact={exact:>5} ({100*exact/n:4.1f}%)")

print()
print("=" * 80)
print("B. Biggest single-decision blowups vs floor (|dev| >= 50), with thinking")
print("=" * 80)
for cfg in ["kb_pointer_gated", "kb_system_gated", "kb_introspect_gated"]:
    runs = load(cfg)
    worst = []
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                opt = order_up_to_order(t["ctx"])
                dev = t.get("order_used") - opt
                if abs(dev) >= 50:
                    conf = t.get("confidence")
                    worst.append((abs(dev), dev, role, t["ctx"].get("t"), opt, t.get("order_used"),
                                  conf if conf is not None else -1,
                                  (t.get("thinking") or t.get("reasoning") or "")[:200]))
    worst.sort(reverse=True)
    print(f"--- {cfg}: {len(worst)} blowups (|dev|>=50), top 4 ---")
    for absd, dev, role, tt, opt, used, conf, th in worst[:4]:
        print(f"  [{role} t={tt}] opt={opt} used={used} dev={dev:+d} conf={conf}")
        print(f"      think: {th}")

print()
print("=" * 80)
print("C. Confidence calibration: does low confidence predict bad decisions?")
print("=" * 80)
for cfg in ["kb_introspect_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    rows = []
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                opt = order_up_to_order(t["ctx"])
                dev = abs(t.get("order_used") - opt)
                rows.append((t.get("confidence"), dev, t.get("gated")))
    low = [r for r in rows if r[0] is not None and r[0] < 0.6]
    high = [r for r in rows if r[0] is not None and r[0] >= 0.6]
    if low:
        print(f"{cfg}: conf<0.6 n={len(low)} mean|dev|={statistics.mean(r[1] for r in low):.1f}  "
              f"conf>=0.6 n={len(high)} mean|dev|={statistics.mean(r[1] for r in high):.1f}")
    else:
        print(f"{cfg}: no low-confidence rows (conf<0.6)")

print()
print("=" * 80)
print("D. Role-level order=0 pathology + gate firing")
print("=" * 80)
for cfg in ["kb_introspect_gated", "kb_system_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    role_stats = collections.defaultdict(lambda: [0, 0])  # [order0_with_backlog, gated]
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                if t.get("order_used") == 0 and (t.get("ctx") or {}).get("backlog", 0) > 0:
                    role_stats[role][0] += 1
                if t.get("gated"):
                    role_stats[role][1] += 1
    print(f"--- {cfg} ---")
    for role in ["retailer", "wholesaler", "distributor", "factory"]:
        z0, g = role_stats[role]
        print(f"  {role:<12} order0-with-backlog: {z0:>4}   gated: {g:>4}")
