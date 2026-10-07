#!/usr/bin/env python3
"""w6fix_analysis.py — WAVE6-FIX gate readout (stdlib only).

Reads results/grpo_w6fix_{a0,a0b,a1,a1_seed2,a2,a2_seed2}/train_log.jsonl and the
historical hpc-umbrella/umbrella-results/results/grpo_{a0_base,a11_seed2}/ logs and
evaluates docs/waves/WAVE6-FIX_SPEC.md §2 (V-gates, H1-H5) and §6.2 (rules R-A..R-D).
Thresholds are the pre-registered ones; a missing arm prints `not run`, never a number.

Usage
  python3 scripts/w6fix_analysis.py [--results results] \
      [--hist hpc-umbrella/umbrella-results/results] \
      [--out results/w6fix_report.md] [--json results/w6fix_report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# Golden first-prompt SHA-256 (tests/fixtures/w6fix_collect_legacy_golden.json; step 0,
# seed 12345, curriculum poisson): A0 recorded from the PRE-change code, A1/A2 from the
# contract prompt at implementation time.
A0_FIRST_PROMPT_SHA = "a61e56c58958011688445c0d8b9ed368ec5010122110b2d9b810b0bc8ec4b370"
A1_FIRST_PROMPT_SHA = "97a3b97f378272d98ee4225cb90c89b0e45064c16dda883e6bbd78d707efdb2f"

ARMS = {  # arm -> (dir, seed-group, historical reference dir)
    "a0": ("grpo_w6fix_a0", 1, "grpo_a0_base"),
    "a0b": ("grpo_w6fix_a0b", 1, "grpo_a0_base"),
    "a1": ("grpo_w6fix_a1", 1, "grpo_a0_base"),
    "a1_seed2": ("grpo_w6fix_a1_seed2", 2, "grpo_a11_seed2"),
    "a2": ("grpo_w6fix_a2", 1, "grpo_a0_base"),
    "a2_seed2": ("grpo_w6fix_a2_seed2", 2, "grpo_a11_seed2"),
}
V0_BAND = (0.81, 0.87)
H1_SUPPORT, H1_FALSIFY = 0.50, 0.20
HI, LO_P100, EARLY_STOP = 0.90, 0.20, 0.02
WINDOWS = {"P25": (20, 25), "P50": (45, 50), "P100": (95, 100)}   # [lo, hi) step indices


# ------------------------------------------------------------------ loading --
def load_log(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    recs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "event" in r or "step" not in r:
            continue
        recs.append(r)
    recs.sort(key=lambda r: r["step"])
    return recs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def by_step(recs, lo, hi, field):
    vals = [r.get(field) for r in recs if lo <= r["step"] < hi]
    return mean(vals)


def window(recs, name, field="contract_parse_rate"):
    """Mean of `field` over the P-window; None if the run is shorter than the window."""
    lo, hi = WINDOWS[name]
    if recs is None or not recs or recs[-1]["step"] < hi - 1:
        return None
    return by_step(recs, lo, hi, field)


def p100(recs):
    """(value, flag): P100 window, or P50 read as P100 when stopped early (< 0.02)."""
    v = window(recs, "P100")
    if v is not None:
        return v, ""
    p50 = window(recs, "P50")
    if p50 is not None and p50 < EARLY_STOP:
        return p50, "stopped early (P50<0.02)"
    return None, "incomplete"


# ------------------------------------------------------------------- analysis --
def analyse(results: Path, hist: Path) -> dict:
    logs = {a: load_log(results / d / "train_log.jsonl") for a, (d, _, _) in ARMS.items()}
    ref = {n: load_log(hist / n / "train_log.jsonl") for n in ("grpo_a0_base", "grpo_a11_seed2")}
    out: dict = {"arms_present": {a: (l is not None) for a, l in logs.items()}}

    # ---- V gates
    v: dict = {}
    a0 = logs["a0"]
    if a0:
        pf = by_step(a0, 0, 50, "parse_failure_rate")
        v["V0"] = {"mean_parse_failure_0_49": pf, "band": V0_BAND,
                   "pass": pf is not None and V0_BAND[0] <= pf <= V0_BAND[1]}
        sha0 = a0[0].get("first_prompt_sha256")
        v["V1_a0"] = {"sha": sha0, "pass": sha0 == A0_FIRST_PROMPT_SHA}
    else:
        v["V0"] = v["V1_a0"] = "not run"
    for a in ("a1", "a1_seed2", "a2", "a2_seed2"):
        l = logs[a]
        v[f"V1_{a}"] = ("not run" if not l else
                        {"sha": l[0].get("first_prompt_sha256"),
                         "pass": l[0].get("first_prompt_sha256") == A1_FIRST_PROMPT_SHA
                         if a in ("a1", "a2") else None,
                         "note": "seed-2 arm: step-0 prompt depends on the seed-2 demand path; "
                                 "no golden SHA, not gated" if "seed2" in a else ""})
    base_clip = statistics.median([r["clip_frac"] for r in a0 if r.get("clip_frac") is not None]) \
        if a0 else None
    for a in ("a1", "a1_seed2", "a2", "a2_seed2"):
        l = logs[a]
        if not l or base_clip is None:
            v[f"V2_{a}"] = "not run"
            continue
        mx = max((r["clip_frac"] for r in l if r.get("clip_frac") is not None), default=None)
        v[f"V2_{a}"] = {"max_clip_frac": mx, "a0_median_clip_frac": base_clip,
                        "pass": mx is not None and mx <= 10 * base_clip}
    v["V3"] = "n/a here: parser goldens are hub tests (pytest tests/test_w6fix_contract.py)"
    out["validity"] = v
    dicts = [x for x in v.values() if isinstance(x, dict)]
    if any(x.get("pass") is False for x in dicts):
        validity_ok = False
    elif any(x == "not run" for x in v.values()):
        validity_ok = "incomplete (an arm is not run)"
    else:
        validity_ok = True
    out["validity_ok"] = validity_ok

    # ---- P windows + step-0 compliance
    P: dict = {}
    for a, l in logs.items():
        if not l:
            P[a] = "not run"
            continue
        p100v, flag = p100(l)
        P[a] = {"P25": window(l, "P25"), "P50": window(l, "P50"), "P100": p100v,
                "P100_flag": flag, "steps0_4": by_step(l, 0, 5, "contract_parse_rate"),
                "parse_lenient_0_4": by_step(l, 0, 5, "parse_lenient_rate"),
                "n_steps": len(l)}
    out["P"] = P

    def g(a, k):
        x = P.get(a)
        return None if not isinstance(x, dict) else x[k]

    # ---- per-seed classification (§6.2), then agreement
    seeds = {1: ("a1", "a2"), 2: ("a1_seed2", "a2_seed2")}
    per_seed: dict = {}
    for s, (a1, a2) in seeds.items():
        A1_50, A2_50, A1_100, A2_100 = g(a1, "P50"), g(a2, "P50"), g(a1, "P100"), g(a2, "P100")
        if None in (A1_50, A2_50, A1_100, A2_100):
            per_seed[s] = "not run"
            continue
        if A1_50 >= HI:
            rule = "R-A"
        elif A2_50 >= HI and A2_50 - A1_50 >= 0.20:
            rule = "R-B"
        elif A1_100 < LO_P100 and A2_100 < LO_P100:
            rule = "R-C"
        else:
            rule = "R-D"
        per_seed[s] = rule
    vals = [x for x in per_seed.values()]
    if "not run" in vals:
        rule_final = "not run (need both seeds of A1 and A2)"
    elif len(set(vals)) == 1:
        rule_final = vals[0]
    else:
        rule_final = "R-D (mixed: seeds disagree)"
    out["rules"] = {"per_seed": per_seed, "final": rule_final}

    # ---- hypotheses
    H: dict = {}

    def both(fn):
        rs = []
        for s, (a1, a2) in seeds.items():
            try:
                rs.append(fn(a1, a2))
            except TypeError:
                rs.append(None)
        if any(r is None for r in rs):
            return "not run"
        return rs[0] if rs[0] == rs[1] else "mixed"

    def h1(a1, a2):
        x = g(a1, "steps0_4")
        return "supports" if x >= H1_SUPPORT else ("falsified" if x < H1_FALSIFY else "neither")
    H["H1"] = both(h1)

    def h2(a1, a2):
        if g(a1, "P50") >= HI:
            return "supports"
        if g(a1, "P100") < 0.50 and g(a2, "P100") >= HI:
            return "falsified"
        return "neither"
    H["H2"] = both(h2)

    def h3(a1, a2):
        if g(a1, "P50") >= HI:
            return "n/a (A1 P50 >= 0.90)"
        if g(a2, "P50") >= g(a1, "P50") + 0.20:
            return "supports"
        if g(a2, "P100") <= g(a1, "P100") + 0.05:
            return "falsified"
        return "neither"
    H["H3"] = both(h3)

    def h4(a1, a2):
        if min(g(a1, "P100"), g(a2, "P100")) >= LO_P100 or max(g(a1, "P100"), g(a2, "P100")) >= LO_P100:
            return "falsified"
        return "supports"
    H["H4"] = both(h4)
    ent = {a: (by_step(l, 45, 50, "order_entropy_bits") if l else None) for a, l in logs.items()}
    H["H5_reported_only"] = {"order_entropy_bits_steps45_49": ent}
    out["H"] = H

    # ---- secondary tables
    sec: dict = {}
    for a, l in logs.items():
        if not l:
            continue
        sec[a] = {f: {"0-4": by_step(l, 0, 5, f), "45-49": by_step(l, 45, 50, f),
                      "95-99": by_step(l, 95, 100, f)}
                  for f in ("contract_exact_rate", "parse_lenient_rate", "loose_json_rate",
                            "order_entropy_bits", "n_orders_gt100", "cost_rows_frac",
                            "value_token_frac", "fmt_adv_share")}
    out["secondary"] = sec

    # ---- paired cost windows vs the historical arm of the same seed group
    cost: dict = {}
    for a, (d, grp, rd) in ARMS.items():
        l, hl = logs[a], ref.get(rd)
        if not l or not hl:
            continue
        row = {}
        for name, (lo, hi) in (("41-50", (40, 50)), ("91-100", (90, 100))):
            xs = {r["step"]: r["mean_episode_cost"] for r in l if lo <= r["step"] < hi
                  and r.get("mean_episode_cost") is not None}
            ys = {r["step"]: r["mean_episode_cost"] for r in hl if lo <= r["step"] < hi
                  and r.get("mean_episode_cost") is not None}
            common = sorted(set(xs) & set(ys))
            if not common:
                continue
            row[name] = {"n": len(common), "arm_mean": mean([xs[k] for k in common]),
                         "hist_mean": mean([ys[k] for k in common]),
                         "arm_median": statistics.median([xs[k] for k in common]),
                         "hist_median": statistics.median([ys[k] for k in common]),
                         "cheaper_share": sum(1 for k in common if xs[k] < ys[k]) / len(common)}
        cost[a] = {"vs": rd, "windows": row}
    out["paired_cost"] = cost
    return out


def fmt(x, d=4):
    if x is None:
        return "—"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def render(res: dict) -> str:
    L = ["# WAVE6-FIX gate readout", ""]
    L.append(f"arms present: {res['arms_present']}")
    L.append("")
    L.append("## V-gates (a probe failing any V-gate is void, not negative)")
    for k, x in res["validity"].items():
        L.append(f"- {k}: {x}")
    L.append(f"- all V-gates ok: **{res['validity_ok']}**")
    L.append("")
    L.append("## contract_parse_rate P-windows")
    L.append("| arm | steps 0-4 | P25 | P50 | P100 | flag | lenient 0-4 | n steps |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a, x in res["P"].items():
        if x == "not run":
            L.append(f"| {a} | not run | | | | | | |")
        else:
            L.append(f"| {a} | {fmt(x['steps0_4'])} | {fmt(x['P25'])} | {fmt(x['P50'])} | "
                     f"{fmt(x['P100'])} | {x['P100_flag']} | {fmt(x['parse_lenient_0_4'])} | {x['n_steps']} |")
    L.append("")
    L.append("## Hypotheses (both seeds must agree, else mixed)")
    for k, x in res["H"].items():
        L.append(f"- {k}: {x}")
    L.append("")
    L.append(f"## Decision rule (§6.2): per-seed {res['rules']['per_seed']} -> **{res['rules']['final']}**")
    L.append("")
    L.append("## Paired mean_episode_cost vs historical (same seed group)")
    for a, x in res["paired_cost"].items():
        for w, y in x["windows"].items():
            L.append(f"- {a} vs {x['vs']} steps {w}: arm mean {fmt(y['arm_mean'], 0)} / "
                     f"hist {fmt(y['hist_mean'], 0)} (median {fmt(y['arm_median'], 0)} / "
                     f"{fmt(y['hist_median'], 0)}), cheaper in {fmt(y['cheaper_share'], 2)} of {y['n']} steps")
    L.append("")
    L.append("## Secondary telemetry (windows 0-4 / 45-49 / 95-99)")
    for a, x in res["secondary"].items():
        L.append(f"- {a}: " + "; ".join(
            f"{f} {fmt(w['0-4'])}/{fmt(w['45-49'])}/{fmt(w['95-99'])}" for f, w in x.items()))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--hist", default="hpc-umbrella/umbrella-results/results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = analyse(Path(a.results), Path(a.hist))
    text = render(res)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
