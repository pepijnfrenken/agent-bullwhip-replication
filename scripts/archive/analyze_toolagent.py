#!/usr/bin/env python3
"""Compare toolagent vs floor/baseline on the step (normal) pattern + mine tool traces.

Reads results/toolagent_step/*.jsonl, computes cost/CV, and classifies the
Python code the model wrote in its run_python calls to answer:
  - Did it rediscover order-up-to/base-stock?
  - Did it simulate / brute-force?
  - Did it bug out (NameError etc.) and fall back?
"""
import json, glob, re, statistics
from collections import Counter
from pathlib import Path

RESULTS = Path("results/toolagent_step")
# reference numbers (from README + local floor runs):
FLOOR_STEP = 3681   # order_up_to(3.0, 0.5), CV 0
BASELINE_MEAN = 29_200_000  # plain LLM baseline (approx, step)
BASELINE_CV = 4.97

def classify(code: str) -> list[str]:
    tags = []
    c = code.lower()
    if re.search(r"max\s*\(\s*0\s*,|target|order.?up.?to|base.?stock|inventory position", c):
        tags.append("order_up_to/base_stock")
    if re.search(r"ema|exp\.?mov|moving.?avg|smooth|forecast|np\.mean|statistics\.mean", c):
        tags.append("forecast/mean")
    if re.search(r"safety.?stock|std|sigma|percentile", c):
        tags.append("safety_stock")
    if re.search(r"random|seed|simulate|monte.?carlo|for .* in range|brute", c):
        tags.append("simulation/brute-force")
    if re.search(r"def .*cost|holding_cost|backlog_cost|total_cost|cost", c):
        tags.append("cost-analysis")
    if re.search(r"nameerror|typeerror|error:", c):
        tags.append("HAD_ERROR")
    if not tags:
        tags.append("other")
    return tags

def main():
    runs = []
    for f in sorted(glob.glob(str(RESULTS / "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: runs.append(json.loads(line))
            except: pass
    if not runs:
        print(f"No runs in {RESULTS}"); return
    # filter error records
    ok = [r for r in runs if r.get("total_cost") is not None]
    err = [r for r in runs if r.get("total_cost") is None]
    print(f"Records: {len(runs)} total, {len(ok)} ok, {len(err)} errors")
    if err:
        print("Errors:", [(e.get('run'), e.get('error','')[:80]) for e in err])
    if not ok:
        return
    costs = [r["total_cost"] for r in ok]
    print(f"\n=== ToolAgent on STEP (normal) pattern ===")
    print(f"Runs: {len(costs)}")
    print(f"Costs: {[round(c,1) for c in costs]}")
    mean = statistics.mean(costs)
    cv = statistics.pstdev(costs)/mean if len(costs)>1 else float('nan')
    print(f"Mean cost: {mean:,.1f}  CV: {cv:.3f}")
    print(f"\n=== Comparison (step pattern) ===")
    print(f"  ToolAgent:      {mean:>12,.0f}  CV {cv:.3f}")
    print(f"  Floor (OUT):    {FLOOR_STEP:>12,}  CV 0.000   <- deterministic order-up-to")
    print(f"  Baseline (LLM): {BASELINE_MEAN:>12,}  CV {BASELINE_CV:.2f}  <- plain LLM (approx)")
    if mean > 0:
        print(f"  ToolAgent vs floor: {mean/FLOOR_STEP*100:.1f}% of floor cost ({(mean-FLOOR_STEP)/FLOOR_STEP*100:+.1f}%)")
    # traces
    print(f"\n=== Tool-call classification (all roles/weeks) ===")
    tag_counter = Counter()
    err_counter = Counter()
    code_examples = []
    for r in ok:
        tt = r.get("tool_traces") or {}
        for role, tlist in tt.items():
            if not tlist:
                continue
            # tlist is a list of per-decision lists of {name, arguments, result}
            for decision in tlist:
                if not isinstance(decision, list):
                    continue
                for tc in decision:
                    if not isinstance(tc, dict):
                        continue
                    args = tc.get("arguments") or {}
                    code = args.get("code", "") if isinstance(args, dict) else ""
                    result = tc.get("result", "")
                    tags = classify(code + "\n" + result)
                    for tag in tags:
                        tag_counter[tag] += 1
                    if "HAD_ERROR" in tags or "error" in result.lower()[:60]:
                        err_counter["tool_error"] += 1
                    if len(code_examples) < 3 and len(code) > 50:
                        code_examples.append((role, code[:400], result[:100]))
    for tag, n in tag_counter.most_common():
        print(f"  {tag}: {n}")
    if err_counter:
        print(f"  tool errors: {err_counter['tool_error']}")
    if code_examples:
        print("\n=== Example tool code (first 3 non-trivial) ===")
        for role, code, result in code_examples:
            print(f"\n--- {role} ---")
            print(code)
            print(f"  => {result[:150]}")

if __name__ == "__main__":
    main()
