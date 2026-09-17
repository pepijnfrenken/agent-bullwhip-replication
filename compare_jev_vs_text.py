#!/usr/bin/env python3
"""compare_jev_vs_text.py — Jev vs the stored text-model decisions, same states.

Text-model orders come from the 2026-09-01 tool-calling run
(results/toolagent_compare/*.jsonl) — NOT re-run, per Pino. Jev runs fresh on the
identical states (run_model_comparison.STATES). The reference column is the
order-up-to anchor the project's floor uses (paper point 3.0/0.5 + the OOS-tuned
3.5/0.2), warm-started from the same order history the agent sees, because these
are mid-game snapshots.

Reports: per-state orders side by side, mean absolute deviation from the tuned
anchor, and the over-order rate (order > 2x anchor) — the bullwhip signature.

Usage:  python3 compare_jev_vs_text.py [--modes choice_grid,choice_mult,reads]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from agent_bullwhip.jev_agent import JevAgent, JevAgentConfig
from run_model_comparison import STATES

TEXT_DIR = Path("results/toolagent_compare")
MODEL_FILES = {
    "glm-5.3-flash": "z-ai__glm-5.3-flash.jsonl",
    "ds-v4-flash": "deepseek__deepseek-v4-flash.jsonl",
    "qwen3.8-flash": "Qwen__Qwen3.8-Flash.jsonl",
}


def anchor(ctx: dict, theta: float, lam: float) -> int:
    """Order-up-to with ES, warm-started by replaying last_orders (same as JevAgent)."""
    f = 0.0
    for q in ctx.get("last_orders") or []:
        f = lam * q + (1 - lam) * f
    f = lam * ctx.get("incoming_last", 0) + (1 - lam) * f
    ip = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
    return int(max(0, round(theta * f - ip)))


def load_text_orders() -> dict[str, dict[int, int | None]]:
    """Keyed by GAME WEEK t (the `state` field holds t, not an index).

    A record with order=None is a failed/absent decision; the last record for a
    given t wins (retries append)."""
    out: dict[str, dict[int, int | None]] = {}
    for label, fname in MODEL_FILES.items():
        path = TEXT_DIR / fname
        per_t: dict[int, int | None] = {}
        if path.exists():
            for line in path.open():
                rec = json.loads(line)
                t = rec.get("state")
                if isinstance(t, int):
                    per_t[t] = rec.get("order")
        out[label] = per_t
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="choice_grid,choice_mult,reads")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    text = load_text_orders()
    n = len(STATES)
    print(f"states: {n}   text models on disk: {', '.join(text)}\n")

    rows = []
    for i, ctx in enumerate(STATES):
        a_tuned = anchor(ctx, 3.5, 0.2)
        a_paper = anchor(ctx, 3.0, 0.5)
        row = {"state": i, "t": ctx.get("t"), "ip": ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0),
               "incoming_last": ctx.get("incoming_last"), "anchor_tuned": a_tuned, "anchor_paper": a_paper}
        for label, per_t in text.items():
            row[label] = per_t.get(ctx.get("t"))
        for mode in modes:
            ag = JevAgent("retailer", JevAgentConfig(mode=mode))
            order = ag.decide(ctx)
            row[f"jev_{mode}"] = order
            row[f"conf_{mode}"] = ag.last_decision_meta.get("confidence")
        rows.append(row)

    # ---- table ----
    heads = ["state", "t", "IP", "in_last", "anchor", "paper"] + list(text) + [f"jev:{m}" for m in modes]
    print(" | ".join(f"{h:>13s}" if h in ("state",) else f"{h:>9s}" for h in heads))
    for r in rows:
        cells = [f"{r['state']:>13d}", f"{r['t']:>9}", f"{r['ip']:>9}", f"{r['incoming_last']:>9}",
                 f"{r['anchor_tuned']:>9}", f"{r['anchor_paper']:>9}"]
        for label in text:
            v = r.get(label)
            cells.append(f"{v if v is not None else '-':>9}")
        for mode in modes:
            v = r.get(f"jev_{mode}")
            c = r.get(f"conf_{mode}")
            cells.append(f"{v}{'' if c is None else f' ({c:.2f})':>4}")
        print(" | ".join(cells))

    # ---- summary ----
    print("\n=== deviation from the tuned anchor (theta=3.5, lam=0.2) ===")
    cols = [(label, [r.get(label) for r in rows]) for label in text]
    cols += [(f"jev:{m}", [r.get(f"jev_{m}") for r in rows]) for m in modes]
    for label, vals in cols:
        got = [(r["anchor_tuned"], v) for r, v in zip(rows, vals) if isinstance(v, int)]
        if not got:
            print(f"  {label:>16s}: no decisions on disk")
            continue
        mad = sum(abs(v - a) for a, v in got) / len(got)
        over = sum(1 for a, v in got if v > 2 * a and a > 0)
        print(f"  {label:>16s}: n={len(got):>2}  mean |order - anchor| = {mad:6.1f}   over-order rate (order > 2x anchor) = {over}/{len(got)}")

    out = Path("results/jev_probe") / "jev_vs_text.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "text_models": list(text), "modes": modes}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
