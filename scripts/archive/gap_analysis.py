#!/usr/bin/env python3
"""Mine decision traces for knowledge gaps — the KB iteration loop.

Reads results/<model>-<config>.jsonl (introspect/kb_introspect configs), finds
the decisions where the LLM was CONFIDENT but WRONG (the knowledge gaps), and
prints a gap report to feed back into PROMPT_KB.md.

A decision is "wrong" when the LLM's order deviates far from the safe
deterministic baseline (order-up-to) at the same state — i.e., the playbook
should have steered it closer.
"""
import json
import glob
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

# Which configs carry traces
TRACE_CONFIGS = ["introspect", "kb_introspect", "kb_introspect_gated"]


def safe_order(ctx: dict) -> float:
    """The order-up-to quantity at the same state (our 'ground truth' reference)."""
    # same formula as OrderUpToAgent (theta=3, lam=0.5, warm start from incoming)
    lam = 0.5
    forecast = lam * ctx.get("incoming_last", 0) + (1 - lam) * ctx.get("incoming_last", 0)
    ip = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
    return max(0.0, 3.0 * forecast - ip)


def main() -> None:
    files = sorted(glob.glob(str(RESULTS / "*-*.jsonl")))
    traces = []
    for f in files:
        tag = Path(f).stem
        if not any(c in tag for c in TRACE_CONFIGS):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            for role, role_traces in (r.get("traces") or {}).items():
                for tr in role_traces:
                    tr["_tag"] = tag
                    tr["_role"] = role
                    traces.append(tr)
    if not traces:
        print("No traces found. Run an introspect/kb_introspect config first.")
        return

    print(f"=== GAP REPORT: {len(traces)} traced decisions ===")
    print()
    gaps = []
    for tr in traces:
        ctx = tr.get("ctx", {})
        safe = safe_order(ctx)
        used = tr.get("order_used")
        raw = tr.get("order")
        if used is None:
            continue
        # a gap = the LLM's chosen order is far from the safe policy (either direction)
        if abs(used - safe) > max(2.0, 0.5 * safe):
            gaps.append({**tr, "safe": safe, "dev": used - safe})

    print(f"Confident-but-wrong decisions (|dev| > max(2, 0.5*safe)): {len(gaps)} / {len(traces)}")
    print()
    # bucket by role
    by_role = Counter(g["_role"] for g in gaps)
    print("By role:", dict(by_role))
    # top triggers: what state did the gaps happen in?
    triggers = Counter()
    for g in gaps:
        ctx = g["ctx"]
        if ctx.get("backlog", 0) > 0:
            triggers["backlog>0"] += 1
        if ctx.get("incoming_now", 0) >= 2 * ctx.get("incoming_last", 1):
            triggers["demand_step"] += 1
        if ctx.get("outstanding", 0) > 0:
            triggers["pipeline_nonempty"] += 1
        conf = g.get("confidence")
        if conf is not None and conf < 0.5:
            triggers["low_conf"] += 1
    print("Trigger buckets:", dict(triggers))

    print()
    print("=== Sample gaps (role | conf | order->used | safe | reasoning) ===")
    for g in gaps[:12]:
        print(f"[{g['_tag']}] {g['_role']:<12} conf={g.get('confidence')} "
              f"order={g.get('order')} used={g.get('order_used')} safe={g['safe']:.0f} "
              f"gated={g.get('gated')} | {g.get('reasoning','')[:80]}")


if __name__ == "__main__":
    sys.exit(main())
