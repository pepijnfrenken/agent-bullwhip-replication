#!/usr/bin/env python3
"""Analyze collected tool-agent traces: mistake taxonomy, error patterns, behavior.

Reads results/toolagent_traces/*.jsonl (one record per decision) and answers:
  1. How often does the model match the floor?
  2. When it deviates, by how much (over/under)?
  3. What tool errors happen (NameError, wrong fn, etc.)?
  4. Does it USE the tool or just answer?
  5. What code patterns does it write (base-stock, simulation, forecast)?
  6. Examples of each mistake class.
"""
import json, glob, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

TRACES = Path("results/toolagent_traces")


def classify_code(code: str) -> list[str]:
    tags = []
    c = code.lower()
    if re.search(r"max\s*\(\s*0\s*,|target|order.?up.?to|base.?stock|inventory position|net inventory", c):
        tags.append("base_stock_formula")
    if re.search(r"ema|exp\.?mov|moving.?avg|smooth|forecast|np\.mean|statistics\.mean|mean\s*\(", c):
        tags.append("forecast/mean")
    if re.search(r"safety.?stock|std|sigma|percentile|service level", c):
        tags.append("safety_stock")
    if re.search(r"for .* in range|brute|simulate|random|monte.?carlo|def .*cost|holding_cost|backlog_cost", c):
        tags.append("simulation/brute")
    if re.search(r"print\s*\(", c):
        tags.append("uses_print")
    if not tags:
        tags.append("other_code")
    return tags


def main():
    recs = []
    for f in sorted(glob.glob(str(TRACES / "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: recs.append(json.loads(line))
            except: pass
    if not recs:
        print(f"No trace records in {TRACES}"); return
    print(f"Total decisions: {len(recs)}")

    # 1. floor match
    matched = sum(1 for r in recs if r.get("floor_order") is not None and r.get("model_order") == r.get("floor_order"))
    parse_fail = sum(1 for r in recs if r.get("model_order") is None)
    print(f"Match floor: {matched} ({matched/len(recs):.1%})")
    print(f"Parse fail (no order): {parse_fail} ({parse_fail/len(recs):.1%})")

    # 2. deviation direction/magnitude
    deltas = [r["delta"] for r in recs if r.get("delta") is not None]
    if deltas:
        over = sum(1 for d in deltas if d > 0)
        under = sum(1 for d in deltas if d < 0)
        print(f"\nDeviation: over-order {over} ({over/len(deltas):.1%}), under-order {under} ({under/len(deltas):.1%})")
        print(f"Mean |delta|: {statistics.mean(abs(d) for d in deltas):.2f} units")
        print(f"Mean delta: {statistics.mean(deltas):+.2f}")
        print(f"Max over: {max(deltas)}, Max under: {min(deltas)}")

    # 3. tool errors
    err_counter = Counter(r.get("tool_error") for r in recs)
    print(f"\nTool error types: {dict(err_counter)}")
    err_rate = (len(recs) - err_counter.get('none', 0)) / len(recs)
    print(f"Tool error rate: {err_rate:.1%}")

    # 4. tool use
    used = sum(1 for r in recs if r.get("tool_calls"))
    print(f"\nUsed tool: {used} ({used/len(recs):.1%})")
    no_tool = [r for r in recs if not r.get("tool_calls")]
    if no_tool:
        print(f"  (no tool call — {len(no_tool)} decisions)")

    # 5. code classification
    code_tags = Counter()
    for r in recs:
        code = r.get("code", "")
        if code:
            for tag in classify_code(code):
                code_tags[tag] += 1
    print(f"\nCode patterns (of {sum(code_tags.values())} tool calls):")
    for tag, n in code_tags.most_common():
        print(f"  {tag}: {n}")

    # 6. examples of each mistake class
    print("\n=== EXAMPLE MISTAKES ===")
    # a) tool error examples
    err_examples = [r for r in recs if r.get("tool_error") != "none"]
    if err_examples:
        print(f"\n--- Tool error examples ({len(err_examples)} total) ---")
        for r in err_examples[:3]:
            print(f"\n[{r['role']} t={r['t']}] err={r['tool_error']} order={r['model_order']} floor={r['floor_order']}")
            print(f"  code: {r['code'][:250]}")
            print(f"  result: {r['exec_result'][:120]}")
    # b) big over-order examples
    over_ex = sorted([r for r in recs if r.get("delta") is not None and r["delta"] > 5], key=lambda r: -r["delta"])
    if over_ex:
        print(f"\n--- Big over-order examples ({len(over_ex)} total >5) ---")
        for r in over_ex[:3]:
            print(f"\n[{r['role']} t={r['t']}] model={r['model_order']} floor={r['floor_order']} delta=+{r['delta']}")
            print(f"  ctx: oh={r['ctx'].get('on_hand')} backlog={r['ctx'].get('backlog')} outstanding={r['ctx'].get('outstanding')} incoming_last={r['ctx'].get('incoming_last')}")
            print(f"  code: {r['code'][:200]}")
            print(f"  result: {r['exec_result'][:100]}")
    # c) under-order examples
    under_ex = sorted([r for r in recs if r.get("delta") is not None and r["delta"] < -5], key=lambda r: r["delta"])
    if under_ex:
        print(f"\n--- Big under-order examples ({len(under_ex)} total <-5) ---")
        for r in under_ex[:3]:
            print(f"\n[{r['role']} t={r['t']}] model={r['model_order']} floor={r['floor_order']} delta={r['delta']}")
            print(f"  ctx: oh={r['ctx'].get('on_hand')} backlog={r['ctx'].get('backlog')} outstanding={r['ctx'].get('outstanding')} incoming_last={r['ctx'].get('incoming_last')}")
            print(f"  code: {r['code'][:200]}")
            print(f"  result: {r['exec_result'][:100]}")

    # 7. mistakes by role
    print("\n=== Match rate by role ===")
    by_role = defaultdict(list)
    for r in recs:
        by_role[r["role"]].append(r)
    for role, rs in by_role.items():
        m = sum(1 for r in rs if r.get("floor_order") is not None and r.get("model_order") == r.get("floor_order"))
        print(f"  {role}: {m}/{len(rs)} ({m/len(rs):.1%})")

    # 8. reasoning availability
    with_reasoning = sum(1 for r in recs if r.get("reasoning"))
    print(f"\n=== Reasoning ===")
    print(f"  decisions with reasoning: {with_reasoning}/{len(recs)} ({with_reasoning/len(recs):.1%})")
    # example reasoning for an error decision
    err_with_r = [r for r in recs if r.get("tool_error") != "none" and r.get("reasoning")]
    if err_with_r:
        print(f"\n  Example reasoning on an error decision:")
        print(f"  {err_with_r[0]['reasoning'][:400]}")


if __name__ == "__main__":
    main()
