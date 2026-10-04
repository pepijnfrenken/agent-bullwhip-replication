#!/usr/bin/env python3
"""eval-analyze.py — wave-6 a0_base eval analysis (stdlib only).

Reads the paired arm JSONLs (untrained vs trained), the floor trajectory/summary
JSONs, and writes the eval-a0 report sections with:
  - per-arm cost stats + parse-failure rate (vs training reference)
  - per-arm order stats + collapse check (near-constant orders / floor correlation)
  - rogue-order breakdown: games with >=1 rogue order (>=1e5) vs clean games,
    total_cost stats with vs without rogue games (sweep-363909 watch item)
  - paired per-run deltas + relative deltas (mean/CV/max)

Usage:
  python3 hpc-umbrella/eval-analyze.py \
      --untrained results/local-eval/untrained-step/qwen3-0.6b-untrained-baseline.jsonl \
      --trained   results/local-eval/trained-step/qwen3-0.6b-trained-baseline.jsonl \
      --pattern step --n-weeks 36 --n-roles 4 \
      --floor-json results/local-eval/floors/floor_trajectories.json \
      --floor-summary results/local-eval/floors/floors_summary.json \
      --training-parse-ref 0.847 \
      --report hpc-umbrella/umbrella-results/eval-a0-report.md [--append]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROLES = ["retailer", "wholesaler", "distributor", "factory"]
ROGUE_THRESHOLD = 1e5  # sweep-363909 watch item: any single order >= 1e5 is a rogue order


def load_arm(path: Path, n_weeks: int, n_roles: int) -> tuple[list[dict], list[dict], int]:
    """Return (complete_records, failed_records, decisions_per_run)."""
    complete, failed = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("total_cost") is None:
                failed.append(rec)
            else:
                complete.append(rec)
    return complete, failed, n_weeks * n_roles


def cost_stats(recs: list[dict]) -> dict:
    costs = [r["total_cost"] for r in recs]
    n = len(costs)
    mean = sum(costs) / n
    var = sum((c - mean) ** 2 for c in costs) / n
    sd = math.sqrt(var)
    return {
        "n": n, "mean": mean, "std": sd, "cv": sd / mean if mean else float("nan"),
        "min": min(costs), "max": max(costs),
    }


def rogue_orders_per_game(recs: list[dict], n_weeks: int) -> list[int]:
    """Rogue-order (>= ROGUE_THRESHOLD) count per completed game, aligned to recs order."""
    return [
        sum(1 for vals in r["orders"].values() for o in vals[:n_weeks] if o >= ROGUE_THRESHOLD)
        for r in recs
    ]


def rogue_breakdown(recs: list[dict], n_weeks: int) -> dict:
    """Rogue-order frequency + total_cost stats with vs without rogue games."""
    counts = rogue_orders_per_game(recs, n_weeks)
    clean = [r for r, c in zip(recs, counts) if c == 0]
    n_rogue_games = len(recs) - len(clean)
    cs_full = cost_stats(recs)
    cs_clean = cost_stats(clean) if clean else None
    return {
        "n_games": len(recs),
        "n_rogue_games": n_rogue_games,
        "rogue_game_freq": n_rogue_games / len(recs) if recs else float("nan"),
        "n_rogue_orders": sum(counts),
        "costs_with_rogue": cs_full,
        "costs_without_rogue": cs_clean,
    }


def order_grid(recs: list[dict], n_weeks: int) -> dict:
    """Flatten orders + per-(role,week) across-run series + per-role stats."""
    orders_by_cell: dict[tuple, list] = {}
    all_orders: list[int] = []
    for r in recs:
        for role in ROLES:
            vals = r["orders"].get(role, [])
            for w, v in enumerate(vals[:n_weeks]):
                orders_by_cell.setdefault((role, w), []).append(v)
                all_orders.append(v)
    per_cell_std = [statistics.pstdev(v) for v in orders_by_cell.values() if len(v) > 1]
    per_cell_mean = [statistics.fmean(v) for v in orders_by_cell.values()]
    # rogue-order watch (sweep 363909 standing item): any single order >= 1e5
    rogue_runs = sum(1 for r in recs for v in r["orders"].values() for o in v[:n_weeks] if o >= 1e5)
    role_stats = {}
    for role in ROLES:
        vals = [v for r in recs for v in r["orders"].get(role, [])[:n_weeks]]
        role_stats[role] = {
            "n_unique": len(set(vals)),
            "mean": statistics.fmean(vals) if vals else float("nan"),
            "std": statistics.pstdev(vals) if vals else float("nan"),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n_rogue_orders": sum(1 for v in vals if v >= 1e5),
        }
    return {
        "all_mean": statistics.fmean(all_orders) if all_orders else float("nan"),
        "all_std": statistics.pstdev(all_orders) if all_orders else float("nan"),
        "all_max": max(all_orders) if all_orders else None,
        "n_rogue_orders": rogue_runs,
        "cell_std_mean": statistics.fmean(per_cell_std) if per_cell_std else float("nan"),
        "cell_std_max": max(per_cell_std) if per_cell_std else float("nan"),
        "cell_std_min": min(per_cell_std) if per_cell_std else float("nan"),
        "cell_std_lt1_frac": (sum(1 for s in per_cell_std if s < 1.0) / len(per_cell_std)) if per_cell_std else float("nan"),
        "cell_mean_min": min(per_cell_mean) if per_cell_mean else float("nan"),
        "cell_mean_max": max(per_cell_mean) if per_cell_mean else float("nan"),
        "role_stats": role_stats,
    }


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb)


def floor_corr(recs: list[dict], traj: dict, pattern: str, floor_name: str, n_weeks: int) -> dict:
    """Correlate arm orders vs floor orders on the paired demand paths."""
    a, b = [], []
    if pattern == "step":
        # all runs share one canonical path -> mean across runs at (role,week)
        cell: dict[tuple, list] = {}
        for r in recs:
            for role in ROLES:
                for w, v in enumerate(r["orders"].get(role, [])[:n_weeks]):
                    cell.setdefault((role, w), []).append(v)
        for (role, w), vals in cell.items():
            a.append(statistics.fmean(vals))
            b.append(traj["0"][floor_name][role][w])
    else:
        # run i pairs with seed 1000+i -> per-(pass,role,week)
        for i, r in enumerate(recs):
            p = str(i)
            if p not in traj or floor_name not in traj[p]:
                continue
            for role in ROLES:
                for w, v in enumerate(r["orders"].get(role, [])[:n_weeks]):
                    a.append(v)
                    b.append(traj[p][floor_name][role][w])
    return {"r": pearson(a, b), "n_points": len(a)}


def parse_failure_rate(recs: list[dict], decisions_per_run: int) -> tuple[float, int, int]:
    fails = sum(sum(r["failures"].values()) for r in recs)
    total = decisions_per_run * len(recs)
    return (fails / total if total else float("nan"), fails, total)


def paired_deltas(u_recs: list[dict], t_recs: list[dict]) -> dict:
    u_by_run = {r["run"]: r for r in u_recs}
    t_by_run = {r["run"]: r for r in t_recs}
    runs = sorted(set(u_by_run) & set(t_by_run))
    deltas = [t_by_run[i]["total_cost"] - u_by_run[i]["total_cost"] for i in runs]
    n = len(deltas)
    mean_d = sum(deltas) / n
    sd_d = statistics.pstdev(deltas) if n > 1 else 0.0
    median_d = statistics.median(deltas)
    trained_worse = sum(1 for d in deltas if d > 0)
    # exact two-sided sign test (binomial, p=0.5): 2 x tail prob of the observed side
    from math import comb
    k = trained_worse
    if k == n // 2 and n % 2 == 0:
        p_val = 1.0
    else:
        tail = sum(comb(n, j) for j in range(k if k > n / 2 else 0, n + 1 if k > n / 2 else k + 1)) / (2 ** n)
        p_val = min(2 * tail, 1.0)
    return {
        "n_pairs": n, "mean_delta": mean_d, "std_delta": sd_d, "median_delta": median_d,
        "min_delta": min(deltas), "max_delta": max(deltas),
        "trained_worse_count": trained_worse, "sign_test_p": p_val,
    }


def relative_deltas(u: dict, t: dict) -> dict:
    out = {}
    for k in ("mean", "cv", "max"):
        if u.get(k):
            out[k] = (t[k] - u[k]) / u[k]
        else:
            out[k] = float("nan")
    return out


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--untrained", required=True, type=Path)
    ap.add_argument("--trained", required=True, type=Path)
    ap.add_argument("--pattern", required=True, choices=["step", "noisy"])
    ap.add_argument("--n-weeks", type=int, default=36)
    ap.add_argument("--n-roles", type=int, default=4)
    ap.add_argument("--floor-json", required=True, type=Path)
    ap.add_argument("--floor-summary", required=True, type=Path)
    ap.add_argument("--training-parse-ref", type=float, default=0.847)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    traj_all = json.loads(args.floor_json.read_text())["trajectories"][args.pattern]
    floor_summ = json.loads(args.floor_summary.read_text())[args.pattern]

    u_recs, u_failed, dec_per_run = load_arm(args.untrained, args.n_weeks, args.n_roles)
    t_recs, t_failed, _ = load_arm(args.trained, args.n_weeks, args.n_roles)

    u_cs, t_cs = cost_stats(u_recs), cost_stats(t_recs)
    u_og, t_og = order_grid(u_recs, args.n_weeks), order_grid(t_recs, args.n_weeks)
    u_rogue, t_rogue = rogue_breakdown(u_recs, args.n_weeks), rogue_breakdown(t_recs, args.n_weeks)
    u_pf, u_fails, u_total = parse_failure_rate(u_recs, dec_per_run)
    t_pf, t_fails, t_total = parse_failure_rate(t_recs, dec_per_run)
    deltas = paired_deltas(u_recs, t_recs)
    rel = relative_deltas(u_cs, t_cs)

    floor_refs = {name: floor_summ[name] for name in floor_summ}
    corr_lines = []
    for name in floor_summ:
        cu = floor_corr(u_recs, traj_all, args.pattern, name, args.n_weeks)
        ct = floor_corr(t_recs, traj_all, args.pattern, name, args.n_weeks)
        corr_lines.append((name, cu, ct))

    # --- collapse verdict ---
    flag = False
    reasons = []
    if u_og["cell_std_mean"] < 1.0 and max(v["n_unique"] for v in u_og["role_stats"].values()) <= 4:
        flag = True
        reasons.append(f"untrained near-constant orders (cell-std mean {u_og['cell_std_mean']:.2f}, "
                       f"n_unique<=4 per role)")
    if t_og["cell_std_mean"] < 1.0 and max(v["n_unique"] for v in t_og["role_stats"].values()) <= 4:
        flag = True
        reasons.append(f"trained near-constant orders (cell-std mean {t_og['cell_std_mean']:.2f}, "
                       f"n_unique<=4 per role)")
    for name, cu, ct in corr_lines:
        if cu["r"] > 0.95:
            flag = True
            reasons.append(f"untrained corr vs {name} floor r={cu['r']:.3f}")
        if ct["r"] > 0.95:
            flag = True
            reasons.append(f"trained corr vs {name} floor r={ct['r']:.3f}")

    # --- stdout ---
    label = args.label or args.pattern
    print(f"\n===== {label} ({args.pattern}) =====")
    print(f"{'arm':<12}{'n':>4}{'mean':>10}{'std':>9}{'cv':>7}{'min':>9}{'max':>9}  parse-fail"
          f"  (ref {args.training_parse_ref})")
    for name, cs, pf in (("untrained", u_cs, u_pf), ("trained", t_cs, t_pf)):
        print(f"{name:<12}{cs['n']:>4}{fmt(cs['mean'],0):>10}{fmt(cs['std'],0):>9}{fmt(cs['cv'],2):>7}"
              f"{fmt(cs['min'],0):>9}{fmt(cs['max'],0):>9}  {pf:.3f} ({u_fails if name=='untrained' else t_fails}"
              f"/{u_total if name=='untrained' else t_total})")
    if u_failed or t_failed:
        print(f"NOTE failed runs: untrained={len(u_failed)} trained={len(t_failed)}"
              f" {[r.get('error','')[:60] for r in u_failed + t_failed]}")
    print(f"order stats: untrained mean={fmt(u_og['all_mean'])} std={fmt(u_og['all_std'])}"
          f" max={u_og['all_max']} rogue={u_og['n_rogue_orders']} | trained mean={fmt(t_og['all_mean'])}"
          f" std={fmt(t_og['all_std'])} max={t_og['all_max']} rogue={t_og['n_rogue_orders']}")
    for nm, rb in (("untrained", u_rogue), ("trained", t_rogue)):
        cs = rb["costs_without_rogue"]
        print(f"rogue games ({nm}): {rb['n_rogue_games']}/{rb['n_games']} "
              f"freq={rb['rogue_game_freq']:.3f} orders={rb['n_rogue_orders']} | "
              f"cost w/ rogue mean={fmt(rb['costs_with_rogue']['mean'],0)} cv={rb['costs_with_rogue']['cv']:.3f} | "
              f"wo/ rogue mean={fmt(cs['mean'],0) if cs else '—'} cv={cs['cv']:.3f} "
              f"max={fmt(cs['max'],0) if cs else '—'} (n={cs['n'] if cs else 0})")
    print(f"per-(agent,week) across-run std: untrained mean={u_og['cell_std_mean']:.2f} "
          f"max={u_og['cell_std_max']:.2f} lt1={u_og['cell_std_lt1_frac']:.2f} | "
          f"trained mean={t_og['cell_std_mean']:.2f} max={t_og['cell_std_max']:.2f} "
          f"lt1={t_og['cell_std_lt1_frac']:.2f}")
    for name in ("retailer", "wholesaler", "distributor", "factory"):
        ur, tr = u_og["role_stats"][name], t_og["role_stats"][name]
        print(f"  {name:<12} untrained n_unique={ur['n_unique']:>2} mean={fmt(ur['mean']):>6} "
              f"max={ur['max']:>4} | trained n_unique={tr['n_unique']:>2} mean={fmt(tr['mean']):>6} "
              f"max={tr['max']:>4}")
    for name, cu, ct in corr_lines:
        print(f"corr vs {name} floor: untrained r={cu['r']:.3f} (n={cu['n_points']}) | "
              f"trained r={ct['r']:.3f} (n={ct['n_points']})")
    print(f"\npaired deltas (trained-untrained, n={deltas['n_pairs']}): mean={fmt(deltas['mean_delta'],0)} "
          f"median={fmt(deltas['median_delta'],0)} std={fmt(deltas['std_delta'],0)} "
          f"min={fmt(deltas['min_delta'],0)} max={fmt(deltas['max_delta'],0)}; "
          f"trained worse in {deltas['trained_worse_count']}/{deltas['n_pairs']} "
          f"(sign-test p={deltas['sign_test_p']:.4f})")
    print(f"relative deltas: mean {rel['mean']:+.3f}  cv {rel['cv']:+.3f}  max {rel['max']:+.3f}")
    if flag:
        print("COLLAPSE FLAG: " + "; ".join(reasons))

    # --- report ---->
    sec = [f"\n## {label} arm ({args.pattern})", ""]
    sec.append(f"**Untrained**: `{args.untrained}` · **Trained**: `{args.trained}` · completed "
               f"{u_cs['n']}/{t_cs['n']} runs · failed {len(u_failed)}/{len(t_failed)}"
               + (" (see note below)" if (u_failed or t_failed) else ""))
    sec.append("")
    sec.append("| arm | n | mean | std | CV | min | max | parse-fail |")
    sec.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, cs, pf in (("untrained", u_cs, u_pf), ("trained", t_cs, t_pf)):
        sec.append(f"| {name} | {cs['n']} | {fmt(cs['mean'],0)} | {fmt(cs['std'],0)} | "
                   f"{fmt(cs['cv'],3)} | {fmt(cs['min'],0)} | {fmt(cs['max'],0)} | {pf:.3f} |")
    sec.append("")
    sec.append(f"Training-time parse-failure reference: {args.training_parse_ref} (mean over 300 GRPO "
               "steps, `train_log.jsonl`).")
    sec.append("")
    sec.append("### Order stats + collapse check")
    sec.append("")
    sec.append("| arm | mean order | std order | max order | per-(agent,week) std (across runs) | "
               "cell-std frac < 1 |")
    sec.append("|---|---:|---:|---:|---:|---:|")
    sec.append(f"| untrained | {fmt(u_og['all_mean'])} | {fmt(u_og['all_std'])} | {u_og['all_max']} | "
               f"{u_og['cell_std_mean']:.2f} (max {u_og['cell_std_max']:.2f}) | {u_og['cell_std_lt1_frac']:.2f} |")
    sec.append(f"| trained | {fmt(t_og['all_mean'])} | {fmt(t_og['all_std'])} | {t_og['all_max']} | "
               f"{t_og['cell_std_mean']:.2f} (max {t_og['cell_std_max']:.2f}) | {t_og['cell_std_lt1_frac']:.2f} |")
    sec.append("")
    sec.append(f"Rogue orders (single order ≥ 1e5, sweep-363909 watch item): untrained "
               f"**{u_og['n_rogue_orders']}**, trained **{t_og['n_rogue_orders']}** across all "
               "runs and roles.")
    sec.append("")
    sec.append("Rogue-order breakdown — games with ≥1 rogue order vs clean games:")
    sec.append("")
    sec.append("| arm | games w/ rogue | rogue freq | n rogue orders | total_cost w/ rogue (n, mean, CV, max) | "
               "total_cost wo/ rogue (n, mean, CV, max) |")
    sec.append("|---|---:|---:|---:|---|---|")
    for nm, rb in (("untrained", u_rogue), ("trained", t_rogue)):
        f_ = rb["costs_with_rogue"]
        c_ = rb["costs_without_rogue"]
        c_cv = (f"{c_['cv']:.3f}") if c_ else "—"
        c_mean = fmt(c_['mean'], 0) if c_ else "—"
        c_max = fmt(c_['max'], 0) if c_ else "—"
        c_n = c_['n'] if c_ else 0
        sec.append(
            f"| {nm} | {rb['n_rogue_games']} | {rb['rogue_game_freq']:.3f} | {rb['n_rogue_orders']} | "
            f"n={f_['n']} mean={fmt(f_['mean'],0)} CV={f_['cv']:.3f} max={fmt(f_['max'],0)} | "
            f"n={c_n} mean={c_mean} CV={c_cv} max={c_max} |")
    sec.append("")
    sec.append("Unique order values per role (untrained → trained):")
    sec.append("")
    sec.append("| role | untrained | trained |")
    sec.append("|---|---:|---:|")
    for name in ROLES:
        ur, tr = u_og["role_stats"][name], t_og["role_stats"][name]
        sec.append(f"| {name} | {ur['n_unique']} | {tr['n_unique']} |")
    sec.append("")
    sec.append("Correlation vs order-up-to floors (same demand paths):")
    sec.append("")
    sec.append("| floor | untrained r | trained r | n points |")
    sec.append("|---|---:|---:|---:|")
    for name, cu, ct in corr_lines:
        sec.append(f"| {name} ({fmt(floor_refs[name]['mean'],0)}) | {cu['r']:.3f} | {ct['r']:.3f} "
                   f"| {cu['n_points']} |")
    if flag:
        sec.append("")
        sec.append(f"**COLLAPSE FLAG:** {'; '.join(reasons)} — a trained model that 'wins' by emitting "
                   "near-constant orders is reported as a distilled-formula regime, not celebrated.")
    sec.append("")
    sec.append("### Paired deltas (trained − untrained, same run index = same demand path)")
    sec.append("")
    sec.append(f"| n pairs | mean Δ | median Δ | std Δ | min Δ | max Δ | trained worse | sign-test p |")
    sec.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    sec.append(f"| {deltas['n_pairs']} | {fmt(deltas['mean_delta'],0)} | {fmt(deltas['median_delta'],0)} | "
               f"{fmt(deltas['std_delta'],0)} | {fmt(deltas['min_delta'],0)} | {fmt(deltas['max_delta'],0)} | "
               f"{deltas['trained_worse_count']} | {deltas['sign_test_p']:.4f} |")
    sec.append("")
    sec.append("Relative deltas: mean **{0:+.1%}**, CV **{1:+.1%}**, max **{2:+.1%}**.".format(
        rel["mean"], rel["cv"], rel["max"]))
    sec.append("")
    if u_failed or t_failed:
        sec.append(f"**Run-failure note:** untrained {len(u_failed)} failed run(s), trained "
                   f"{len(t_failed)} — excluded from cost stats; errors: "
                   f"{[r.get('error', '')[:80] for r in u_failed + t_failed]}.")
        sec.append("")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.append and args.report.exists()) else "w"
    with args.report.open(mode) as f:
        f.write("\n".join(sec))


if __name__ == "__main__":
    main()