#!/usr/bin/env python3
"""wave7_analysis.py — wave-7 gate readout from the protocol evals (stdlib only).

Reads
  results/eval/w7-<arm>-<mode>-<pattern>.jsonl   the 21 protocol arms (decision_eval schema)
  results/eval/w7-<arm>-perm.json                AUDIT-1 amendment-1 permutation-gate probes
  results/wave7_floors.json                      the 30-pass floors (pass i == eval run i; pass -1 = the n=1 step game)
  results/local-eval/trained-{step,noisy}/...    wave-6 a0_base trained arm (C3 reference)

Prints (markdown) and optionally writes
  - per-arm cost stats, parse failures, rogue/edge/collapse hygiene
  - C1 trained vs its own untrained, paired (median reduction + wins + one-sided sign p)
  - C2 absolute level on the step arm (12.8k = strong, 6.4k = headline)
  - C3 vs the wave-6 a0_base trained arm on identical paths (rogue-aware)
  - C4 3-seed agreement + parse-fail == 0
  - R1 sft vs rlcd paired
  - permutation-gate table (>= 0.90 invariant under 3 shuffled grids on the trained arms)
  - floors context (wf_tuned / untuned / oracle)

Usage
  python3 wave7_analysis.py [--eval-dir results/eval] [--floors results/wave7_floors.json]
      [--c3-step results/local-eval/trained-step/qwen3-0.6b-trained-baseline.jsonl]
      [--c3-noisy results/local-eval/trained-noisy/qwen3-0.6b-trained-baseline.jsonl]
      [--out results/wave7_report.md] [--json results/wave7_stats.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

MODES = ("argmax", "sampled")
PATTERNS = ("step", "noisy")
ARMS = ("untrained", "sft-seed0", "sft-seed1", "sft-seed2",
        "rlcd-seed0", "rlcd-seed1", "rlcd-seed2")
C2_STRONG, C2_HEADLINE = 12_800.0, 6_400.0     # 4x / 2x the step wf floor (3,206)
ROGUE = 1e5                                    # sweep-363909 watch item
EDGE_LO, EDGE_HI = 0, 40                        # option grid 0..40 step 2


# ---------------------------------------------------------------- loading ----
def parse_name(stem: str) -> tuple[str, str, str] | None:
    """w7-<arm>-<mode>-<pattern> -> (arm, mode, pattern); arm may contain '-'."""
    if not stem.startswith("w7-"):
        return None
    parts = stem[3:].split("-")
    if len(parts) < 3:
        return None
    pattern, mode, arm = parts[-1], parts[-2], "-".join(parts[:-2])
    if pattern not in PATTERNS or mode not in MODES:
        return None
    return arm, mode, pattern


def load_arm(path: Path) -> dict:
    complete, failed = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("total_cost") is None or not r.get("orders"):
                failed.append(r)
            else:
                complete.append(r)
    return {"complete": complete, "failed": failed, "path": str(path)}


def load_eval_dir(d: Path) -> dict[tuple[str, str, str], dict]:
    arms = {}
    for p in sorted(d.glob("w7-*.jsonl")):
        key = parse_name(p.stem)
        if key is None:
            continue
        arms[key] = load_arm(p)
    return arms


def load_floors(path: Path) -> dict[tuple[str, int], dict]:
    """{(pattern, pass): record}; the n=1 step game is key (step, -1)."""
    out = {}
    if not path.exists():
        return out
    for r in json.loads(path.read_text()):
        out[(r["pattern"], r["pass"])] = r
    return out


def load_perm(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("w7-*-perm.json")):
        arm = p.stem[3:-5]                      # w7-<arm>-perm
        try:
            out[arm] = json.loads(p.read_text())
        except json.JSONDecodeError:
            out[arm] = {"error": "unreadable"}
    return out


# ------------------------------------------------------------- statistics ----
def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def sign_p(wins: int, n: int) -> float:
    """One-sided exact binomial p (H0: p=0.5), P(X >= wins)."""
    if n <= 0:
        return float("nan")
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def paired(trained: dict[int, float], ref: dict[int, float]) -> dict:
    idx = sorted(set(trained) & set(ref))
    deltas = [trained[i] - ref[i] for i in idx]
    rel = [(trained[i] - ref[i]) / ref[i] for i in idx if ref[i]]
    wins = sum(1 for d in deltas if d < 0)
    n = len(deltas)
    out = {"n": n, "wins": wins, "median_delta": median(deltas) if deltas else float("nan"),
           "mean_delta": statistics.mean(deltas) if deltas else float("nan"),
           "median_rel": median(rel) if rel else float("nan"),
           "sign_p": sign_p(wins, n)}
    if n >= 2:
        sd = statistics.stdev(deltas)
        se = sd / n ** 0.5
        out.update(std_delta=sd, se=se, t=(statistics.mean(deltas) / se) if se else float("inf"))
    return out


def cost_stats(costs: list[float]) -> dict:
    if not costs:
        return {"n": 0}
    m = statistics.mean(costs)
    sd = statistics.stdev(costs) if len(costs) >= 2 else 0.0
    return {"n": len(costs), "mean": m, "median": median(costs), "min": min(costs),
            "max": max(costs), "std": sd, "cv": sd / m if m else float("nan"),
            "se": sd / len(costs) ** 0.5 if len(costs) >= 2 else 0.0}


def orders_of(rec: dict) -> list[float]:
    out = []
    for role_orders in (rec.get("orders") or {}).values():
        if isinstance(role_orders, list):
            out.extend(float(x) for x in role_orders)
        elif isinstance(role_orders, (int, float)):
            out.append(float(role_orders))
    return out


def hygiene(arm: dict) -> dict:
    recs = arm["complete"]
    orders, per_run_max, rogue_runs = [], [], 0
    pf = 0
    for r in recs + arm["failed"]:
        fails = r.get("failures")
        if isinstance(fails, dict):
            pf += sum(int(v) for v in fails.values())
        else:
            pf += int(fails or 0)
    for r in recs:
        o = orders_of(r)
        orders.extend(o)
        if o:
            per_run_max.append(max(o))
            if max(o) >= ROGUE:
                rogue_runs += 1
    edge_hits = sum(1 for o in orders if o in (EDGE_LO, EDGE_HI))
    return {
        "parse_fail_runs": len(arm["failed"]),
        "parse_fail_decisions": pf,
        "n_orders": len(orders),
        "unique_orders": len(set(orders)),
        "max_order": max(orders) if orders else float("nan"),
        "edge_hits": edge_hits,
        "edge_frac": (edge_hits / len(orders)) if orders else float("nan"),
        "rogue_runs": rogue_runs,
    }


def run_costs(arm: dict) -> dict[int, float]:
    return {int(r["run"]): float(r["total_cost"]) for r in arm["complete"]}


def run_max_order(rec: dict) -> float:
    o = orders_of(rec)
    return max(o) if o else 0.0


# --------------------------------------------------------------- reporting ---
def fmt(x: float, nd: int = 0) -> str:
    if x != x:
        return "n/a"
    return f"{x:,.{nd}f}"


def gate_c1(pa: dict) -> dict:
    """C1 needs BOTH reduction variants >= 15% AND wins >= 20/30 AND sign p < 0.02."""
    red1 = 1 - (pa["median_trained"] / pa["median_untrained"]) if pa["median_untrained"] else float("nan")
    red2 = -pa["median_rel"]                    # median of per-path reductions
    need = max(1, math.ceil(0.667 * pa["n"])) if pa["n"] < 30 else 20   # 20/30 wording; scaled below 30 pairs
    ok = (red1 >= 0.15) and (red2 >= 0.15) and (pa["wins"] >= need) and (pa["sign_p"] < 0.02)
    return {"reduction_of_medians": red1, "median_of_reductions": red2,
            "wins_needed": need, "pass": bool(pa["n"] and ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description="wave-7 gate readout (stdlib only)")
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--floors", default="results/wave7_floors.json")
    ap.add_argument("--c3-step", default="results/local-eval/trained-step/qwen3-0.6b-trained-baseline.jsonl")
    ap.add_argument("--c3-noisy", default="results/local-eval/trained-noisy/qwen3-0.6b-trained-baseline.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    edir = Path(args.eval_dir)
    arms = load_eval_dir(edir)
    floors = load_floors(Path(args.floors))
    perms = load_perm(edir)
    stats: dict = {"eval_dir": str(edir), "arms": {}, "c1": {}, "c2": {}, "c3": {},
                   "c4": {}, "r1": {}, "perm": {}, "floors": {}}

    L: list[str] = []
    L.append("# Wave-7 gate readout")
    L.append("")
    L.append(f"eval dir: `{edir}` · arms found: **{len(arms)}** of 21 · "
             f"perm probes: **{len(perms)}** · floors: `{args.floors}`")
    missing = [f"w7-{a}-{m}-{p}" for a in ARMS for m in MODES for p in PATTERNS
               if (a, m, p) not in arms]
    if missing:
        L.append("")
        L.append(f"missing arms ({len(missing)}): " + ", ".join(f"`{m}`" for m in missing))
    L.append("")

    # --- per-arm stats ------------------------------------------------------
    L.append("## Per-arm stats")
    L.append("")
    L.append("| arm | mode | pattern | runs | fail | median | mean | CV | uniq orders | edge % | rogue |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (arm, mode, pattern), a in sorted(arms.items()):
        cs = cost_stats([float(r["total_cost"]) for r in a["complete"]])
        hy = hygiene(a)
        stats["arms"][f"{arm}|{mode}|{pattern}"] = {"cost": cs, "hygiene": hy}
        L.append(f"| {arm} | {mode} | {pattern} | {cs.get('n', 0)} | {hy['parse_fail_runs']} "
                 f"| {fmt(cs.get('median', float('nan')))} | {fmt(cs.get('mean', float('nan')))} "
                 f"| {cs.get('cv', float('nan')):.2f} | {hy['unique_orders']} "
                 f"| {100 * hy['edge_frac']:.1f} | {hy['rogue_runs']} |")
    L.append("")

    # --- C1: trained vs its own untrained ----------------------------------
    L.append("## C1 — trained readout vs its own untrained readout (paired)")
    L.append("")
    L.append("Gate: median reduction >= 15% (both variants) AND >= 20/30 wins AND sign p < 0.02. "
             "Per AUDIT-1 fix 5 the inferential read is the **noisy n=30** arm; the step-sampled "
             "arm is the wave-6 T=0.7 counterpart; the step-argmax arm is n=1 (single labeled cost). "
             "For arms with fewer than 30 pairs the win threshold scales as ceil(2n/3) (deviation "
             "recorded, never silent).")
    L.append("")
    L.append("| trained arm | mode | pattern | n | med trained | med untrained | red1 (medians) "
             "| red2 (median of paths) | wins | sign p | C1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (arm, mode, pattern) in sorted(arms):
        if arm == "untrained" or not arm.startswith(("sft", "rlcd")):
            continue
        ref = arms.get(("untrained", mode, pattern))
        if ref is None:
            continue
        t_costs = run_costs(arms[(arm, mode, pattern)])
        u_costs = run_costs(ref)
        if not t_costs or not u_costs:
            continue
        pa = paired(t_costs, u_costs)
        pa["median_trained"] = median(list(t_costs.values()))
        pa["median_untrained"] = median(list(u_costs.values()))
        g = gate_c1(pa)
        key = f"{arm}|{mode}|{pattern}"
        stats["c1"][key] = {**{k: v for k, v in pa.items() if k != "t"}, **g}
        L.append(f"| {arm} | {mode} | {pattern} | {pa['n']} | {fmt(pa['median_trained'])} "
                 f"| {fmt(pa['median_untrained'])} | {100 * g['reduction_of_medians']:.1f}% "
                 f"| {100 * g['median_of_reductions']:.1f}% | {pa['wins']}/{pa['n']} "
                 f"| {pa['sign_p']:.2e} | {'PASS' if g['pass'] else 'fail'} |")
    L.append("")

    # --- C2: absolute level on step ----------------------------------------
    L.append("## C2 — absolute level (step arm)")
    L.append("")
    L.append(f"WF step floor 3,206 → strong tier <= {fmt(C2_STRONG)} (4x), headline <= {fmt(C2_HEADLINE)} (2x).")
    L.append("")
    L.append("| arm | step-argmax (n=1) | vs 12.8k | step-sampled median (n=30) | vs 12.8k | x floor (sampled) |")
    L.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        a1 = arms.get((arm, "argmax", "step"))
        a30 = arms.get((arm, "sampled", "step"))
        if a1 is None and a30 is None:
            continue
        c1 = cost_stats([float(r["total_cost"]) for r in a1["complete"]]) if a1 else {"median": float("nan")}
        c30 = cost_stats([float(r["total_cost"]) for r in a30["complete"]]) if a30 else {"median": float("nan")}
        tier1 = "headline" if c1["median"] <= C2_HEADLINE else ("strong" if c1["median"] <= C2_STRONG else "over")
        tier30 = "headline" if c30["median"] <= C2_HEADLINE else ("strong" if c30["median"] <= C2_STRONG else "over")
        ratio = f"{c30['median'] / 3206:.1f}x" if c30["median"] == c30["median"] else "n/a"
        stats["c2"][arm] = {"step_argmax": c1["median"], "step_sampled_median": c30["median"],
                            "tier_argmax": tier1, "tier_sampled": tier30}
        L.append(f"| {arm} | {fmt(c1['median'])} | {tier1} | {fmt(c30['median'])} | {tier30} | {ratio} |")
    L.append("")

    # --- C3: vs wave-6 a0_base ---------------------------------------------
    L.append("## C3 — vs the wave-6 a0_base trained arm (identical paths, paired)")
    L.append("")
    L.append("Known deviations (reported, not silently fixed): the wave-6 eval band is **step n=30, "
             "noisy n=10**, so the 20/30-wins wording only binds on the step arm; and the wave-6 arm "
             "has rogue orders (>= 1e5) that dominate its noisy costs, so a rogue-aware subset is "
             "reported next to the raw pairing.")
    L.append("")
    for pat, ref_path in (("step", args.c3_step), ("noisy", args.c3_noisy)):
        rp = Path(ref_path)
        if not rp.exists():
            L.append(f"- {pat}: reference missing (`{rp}`)")
            continue
        ref_arm = load_arm(rp)
        ref_costs = run_costs(ref_arm)
        ref_rogue = {int(r["run"]) for r in ref_arm["complete"] if run_max_order(r) >= ROGUE}
        L.append(f"### {pat} (wave-6 ref: {len(ref_costs)} runs; rogue runs: {sorted(ref_rogue)})")
        L.append("")
        L.append("| arm | mode | n shared | wins | med Δ | sign p | clean n | clean wins | clean med Δ |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for (arm, mode, pattern) in sorted(arms):
            if pattern != pat:
                continue
            t = run_costs(arms[(arm, mode, pattern)])
            pa = paired(t, ref_costs)
            clean_idx = {i: v for i, v in t.items()
                         if i in ref_costs and i not in ref_rogue}
            pac = paired(clean_idx, {i: ref_costs[i] for i in clean_idx})
            stats["c3"][f"{arm}|{mode}|{pat}"] = {"all": {k: v for k, v in pa.items() if k != "t"},
                                                  "clean": {k: v for k, v in pac.items() if k != "t"},
                                                  "ref_rogue_runs": sorted(ref_rogue)}
            L.append(f"| {arm} | {mode} | {pa['n']} | {pa['wins']}/{pa['n']} | {fmt(pa['median_delta'])} "
                     f"| {pa['sign_p']:.2e} | {pac['n']} | {pac['wins']}/{pac['n']} "
                     f"| {fmt(pac['median_delta'])} |")
        L.append("")

    # --- C4: seed agreement -------------------------------------------------
    L.append("## C4 — reproducibility (3-seed agreement, parse-fail == 0)")
    L.append("")
    L.append("Agreement rule: every pair of per-seed means within 2 sqrt(SE_i^2 + SE_j^2) (two-sample z<2).")
    L.append("")
    L.append("| stage | mode | pattern | per-seed medians | z<=2 all pairs | parse-fail total |")
    L.append("|---|---|---|---|---|---|")
    for stage in ("sft", "rlcd"):
        for mode in MODES:
            for pattern in PATTERNS:
                got = [(s, arms.get((f"{stage}-seed{s}", mode, pattern))) for s in (0, 1, 2)]
                got = [(s, a) for s, a in got if a and a["complete"]]
                if not got:
                    continue
                meds = [cost_stats([float(r["total_cost"]) for r in a["complete"]])["median"] for _, a in got]
                cs = [cost_stats([float(r["total_cost"]) for r in a["complete"]]) for _, a in got]
                ok = True
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        se_i, se_j = cs[i]["se"] or 0.0, cs[j]["se"] or 0.0
                        combined = 2 * math.sqrt(se_i ** 2 + se_j ** 2)
                        if abs(cs[i]["mean"] - cs[j]["mean"]) > max(combined, 1e-9):
                            ok = False
                pf = sum(hygiene(a)["parse_fail_runs"] for _, a in got)
                key = f"{stage}|{mode}|{pattern}"
                stats["c4"][key] = {"per_seed_medians": meds, "agree": ok, "parse_fail": pf}
                L.append(f"| {stage} | {mode} | {pattern} | "
                         + " / ".join(fmt(m) for m in meds) + f" | {'yes' if ok else 'NO'} | {pf} |")
    L.append("")

    # --- R1: sft vs rlcd ----------------------------------------------------
    L.append("## R1 — SFT vs RLCD (paired, same seed)")
    L.append("")
    L.append("| seed | mode | pattern | n | wins (rlcd better) | med Δ (rlcd - sft) | sign p |")
    L.append("|---|---|---|---|---|---|---|")
    for s in (0, 1, 2):
        for mode in MODES:
            for pattern in PATTERNS:
                a_sft = arms.get((f"sft-seed{s}", mode, pattern))
                a_rlcd = arms.get((f"rlcd-seed{s}", mode, pattern))
                if not (a_sft and a_rlcd and a_sft["complete"] and a_rlcd["complete"]):
                    continue
                pa = paired(run_costs(a_rlcd), run_costs(a_sft))
                stats["r1"][f"{s}|{mode}|{pattern}"] = {k: v for k, v in pa.items() if k != "t"}
                L.append(f"| {s} | {mode} | {pattern} | {pa['n']} | {pa['wins']}/{pa['n']} "
                         f"| {fmt(pa['median_delta'])} | {pa['sign_p']:.2e} |")
    L.append("")

    # --- permutation gate ---------------------------------------------------
    L.append("## Permutation gate (AUDIT-1 amendment 1)")
    L.append("")
    L.append("Trained arms: >= 0.90 of 512 states keep the same argmax VALUE under 3 shuffled option "
             "grids. Untrained is report-only (low invariance is the expected contrast).")
    L.append("")
    L.append("| arm | states | invariance | gate | pass | min mass |")
    L.append("|---|---|---|---|---|---|")
    for arm, pr in sorted(perms.items()):
        if "error" in pr:
            L.append(f"| {arm} | | | | unreadable | |")
            continue
        stats["perm"][arm] = {k: pr[k] for k in
                              ("n_states", "invariance_rate", "gate", "pass", "min_mass")
                              if k in pr}
        L.append(f"| {arm} | {pr.get('n_states', '')} | {pr.get('invariance_rate', float('nan')):.3f} "
                 f"| {pr.get('gate', 0):.2f} | "
                 f"{'PASS' if pr.get('pass') else ('fail' if pr.get('gate', 0) > 0 else 'control')} "
                 f"| {pr.get('min_mass', float('nan')):.4f} |")
    L.append("")

    # --- floors context -----------------------------------------------------
    if floors:
        L.append("## Floors context (30-pass, results/wave7_floors.json)")
        L.append("")
        L.append("| pattern | passes | wf_tuned median | untuned median | oracle median |")
        L.append("|---|---|---|---|---|")
        for pat in PATTERNS:
            recs = [r for (p, i), r in floors.items() if p == pat and i >= 0]
            if not recs:
                continue
            wf = median([r["wf_cost"] for r in recs])
            un = median([r["untuned_cost"] for r in recs])
            orc = median([r["oracle_cost"] for r in recs])
            stats["floors"][pat] = {"n": len(recs), "wf": wf, "untuned": un, "oracle": orc}
            L.append(f"| {pat} | {len(recs)} | {fmt(wf)} | {fmt(un)} | {fmt(orc)} |")
        st1 = floors.get(("step", -1))
        if st1:
            L.append("")
            L.append(f"n=1 step game: wf {fmt(st1['wf_cost'])} · untuned {fmt(st1['untuned_cost'])} "
                     f"· oracle {fmt(st1['oracle_cost'])}")
    L.append("")

    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(stats, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
