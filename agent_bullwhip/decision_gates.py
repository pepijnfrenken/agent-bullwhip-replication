"""WAVE8 gate math (docs/waves/WAVE8_SPEC.md §6): torch-free, stdlib only.

Everything here is a pure function over plain lists/dicts so the hub can test the
exact thresholds on synthetic AND real (wave-7) shapes; the scripts
(`scripts/wave8_gates.py`, `wave8_analysis.py`, `wave8_grid_probe.py`, ...) only do I/O.
General in the number of labels n (21 for wave 7, 26 for w8_g26).

Conventions fixed here (spec-silent details, logged in WAVE8_SPEC changes log):
  * percentile(q) = nearest rank on the sorted values: v[ceil(q*n)-1] (q in (0,1]).
  * Wilson interval z = 1.96; bootstrap B = 1000, percentile q = 0.95, seed 20261007.
  * relative regret is regret / MEAN reference min-J (a constant), so p90 is the p90
    of per-state regret over that constant.
"""
from __future__ import annotations

import math
import random
import statistics

Z95 = 1.96
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20261007


# --------------------------------------------------------------------------- #
# basic statistics
# --------------------------------------------------------------------------- #
def entropy_nats(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: sorted(values)[ceil(q*n) - 1]."""
    if not values:
        return float("nan")
    v = sorted(values)
    k = max(1, min(len(v), math.ceil(q * len(v))))
    return v[k - 1]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def modal_share(items: list) -> float:
    """Share of the most common item."""
    if not items:
        return float("nan")
    counts: dict = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.values()) / len(items)


def wilson_upper(k: int, n: int, z: float = Z95) -> float:
    """Upper bound of the Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return float("nan")
    ph = k / n
    den = 1 + z * z / n
    centre = ph + z * z / (2 * n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / den)


def argmax_index(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def mutual_information(P: list[list[float]]) -> float:
    """I = H(mean_s pi_s) - mean_s H(pi_s): information the answer carries about the
    state (~0 for a state-blind letter prior, up to log(#distinct answers))."""
    if not P:
        return float("nan")
    n, k = len(P), len(P[0])
    mean = [sum(row[a] for row in P) / n for a in range(k)]
    return entropy_nats(mean) - sum(entropy_nats(row) for row in P) / n


def bootstrap_mi_percentile(P: list[list[float]], *, B: int = BOOTSTRAP_B, q: float = 0.95,
                            seed: int = BOOTSTRAP_SEED) -> float:
    """q-th percentile of I over B bootstrap resamples (rows with replacement)."""
    n, k = len(P), len(P[0])
    rng = random.Random(seed)
    ent = [entropy_nats(r) for r in P]
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        mean = [0.0] * k
        h = 0.0
        for i in idx:
            row = P[i]
            h += ent[i]
            for a in range(k):
                mean[a] += row[a]
        mean = [m / n for m in mean]
        vals.append(entropy_nats(mean) - h / n)
    return percentile(vals, q)


# --------------------------------------------------------------------------- #
# collapse gate (W8 §6.2)
# --------------------------------------------------------------------------- #
def freeze_control_stats(untrained_P: list[list[float]], oracle_argmin: list[int]) -> dict:
    """Thresholds frozen BEFORE training from the untrained readout and the simulator
    oracle on S_nom: H_lo/H_hi (5th/95th percentile of per-state entropy), m_star (modal
    share of the oracle argmin letter), m_hi (its Wilson upper bound), I0 and I_hi0 (95th
    percentile of the untrained readout's state-dependence over bootstrap resamples)."""
    H = [entropy_nats(p) for p in untrained_P]
    n = len(oracle_argmin)
    counts: dict = {}
    for a in oracle_argmin:
        counts[a] = counts.get(a, 0) + 1
    modal_letter, modal_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    argmax_letters = [argmax_index(p) for p in untrained_P]
    return {
        "n_states": len(untrained_P), "n_labels": len(untrained_P[0]),
        "H_lo": percentile(H, 0.05), "H_hi": percentile(H, 0.95),
        "H_median_untrained": median(H),
        "m_star": modal_n / n, "oracle_modal_letter": modal_letter,
        "m_hi": wilson_upper(modal_n, n),
        "I0": mutual_information(untrained_P),
        "I_hi0": bootstrap_mi_percentile(untrained_P),
        "untrained_modal_argmax_share": modal_share(argmax_letters),
        "rule": "collapsed iff [median H < H_lo AND (modal argmax share > m_hi OR I <= I_hi0)] "
                "OR [median H > H_hi AND I <= I_hi0]",
        "bootstrap": {"B": BOOTSTRAP_B, "q": 0.95, "seed": BOOTSTRAP_SEED},
    }


def collapse_gate(P: list[list[float]], frozen: dict) -> dict:
    """Is a trained readout collapsed? Low entropy alone is NOT collapse (a confident
    state-dependent policy is the goal); diffuse-and-state-blind is the other failure."""
    H = [entropy_nats(p) for p in P]
    med_h = median(H)
    share = modal_share([argmax_index(p) for p in P])
    I = mutual_information(P)
    low = med_h < frozen["H_lo"] and (share > frozen["m_hi"] or I <= frozen["I_hi0"])
    high = med_h > frozen["H_hi"] and I <= frozen["I_hi0"]
    reasons = []
    if low:
        reasons.append("low-entropy: " + ("modal argmax share > m_hi" if share > frozen["m_hi"]
                                          else "state-blind (I <= I_hi0)"))
    if high:
        reasons.append("diffuse and state-blind")
    return {"collapsed": bool(low or high), "median_entropy": med_h,
            "modal_argmax_share": share, "I": I, "n_states": len(P),
            "branch": "low" if low else ("high" if high else None), "reasons": reasons}


def consecutive_trips(reads: list[dict], k: int = 2) -> bool:
    return len(reads) >= k and all(bool(r.get("collapsed")) for r in reads[-k:])


# --------------------------------------------------------------------------- #
# permutation gate
# --------------------------------------------------------------------------- #
def permutation_gate(values_by_grid: list[list[int]], gate: float = 0.90) -> dict:
    """values_by_grid[k][s] = argmax VALUE of state s under grid k (k=0 canonical).
    Invariant state = same value under every grid. Same definition as
    scripts/probe_permutation_invariance.py (wave 7 read 0.000 on every arm)."""
    n = len(values_by_grid[0])
    inv = sum(1 for s in range(n) if len({v[s] for v in values_by_grid}) == 1)
    rate = inv / n if n else 0.0
    return {"n_states": n, "n_grids": len(values_by_grid), "invariant": inv,
            "invariance_rate": rate, "gate": gate,
            "pass": bool(rate >= gate) if gate > 0 else True}


# --------------------------------------------------------------------------- #
# pre-flight F0 / F1
# --------------------------------------------------------------------------- #
def f0_stats(J_ref_rows: list[list[float]], grid_values: list[int],
             ref_values: list[int], tau_scale: float = 0.5) -> dict:
    """J over the REFERENCE grid per state -> choice-set quality of `grid_values`
    (a subset of ref_values): relative regret (vs the reference optimum), target mass
    on the top letter (softmax-minus-J target as the labeler builds it), target-argmax
    share at the top letter, and the wave-7-style edge shares."""
    from .decision_labeler import softmax_minus_j
    pos = {v: i for i, v in enumerate(ref_values)}
    cols = [pos[v] for v in grid_values]
    n = len(J_ref_rows)
    ref_min = [min(r) for r in J_ref_rows]
    grid_J = [[r[c] for c in cols] for r in J_ref_rows]
    regret = [min(g) - m for g, m in zip(grid_J, ref_min)]
    denom = sum(ref_min) / n
    rel = sorted(x / denom for x in regret)
    top = len(grid_values) - 1
    p_top, top_arg, zero_arg = [], 0, 0
    for g in grid_J:
        p, _, _ = softmax_minus_j(g, tau_scale)
        p_top.append(p[top])
        a = min(range(len(g)), key=g.__getitem__)
        top_arg += a == top
        zero_arg += a == 0
    return {"n_states": n,
            "regret_mean_rel": (sum(regret) / n) / denom,
            "regret_p90_rel": rel[min(n - 1, math.ceil(0.9 * n) - 1)],
            "top_mass": sum(p_top) / n, "top_argmax_share": top_arg / n,
            "zero_argmax_share": zero_arg / n,
            "ref_argmin_median": median([ref_values[min(range(len(r)), key=r.__getitem__)]
                                         for r in J_ref_rows])}


F0_THRESHOLDS = {  # W8 §6.1 F0
    "mixed": {"regret_mean_rel": 0.005, "regret_p90_rel": 0.02, "top_mass": 0.06,
              "top_argmax_share": 0.08},
    "nominal": {"regret_mean_rel": 0.005, "regret_p90_rel": 0.02, "top_mass": 0.03,
                "top_argmax_share": 0.03},
}


def f0_gate(stats: dict[str, dict]) -> dict:
    """stats = {"mixed": f0_stats(...), "nominal": f0_stats(...)}; each metric must be <= its
    threshold (regret mean and p90 are relative fractions; 0.005 = 0.5 %)."""
    out, ok = {}, True
    for regime, thr in F0_THRESHOLDS.items():
        if regime not in stats:
            return {"pass": None, "reason": f"missing regime {regime}"}
        rows = {}
        for k, t in thr.items():
            v = stats[regime][k]
            rows[k] = {"value": v, "threshold": t, "pass": bool(v <= t)}
            ok &= v <= t
        out[regime] = rows
    return {"pass": bool(ok), "regimes": out}


def f1_gate(pi1_costs: list[float], floor_costs: list[float]) -> dict:
    """Teacher check: closed-loop PI1 must not cost more than its own continuation floor
    (paired on the same paths; mean cost, wins reported)."""
    n = len(pi1_costs)
    wins = sum(1 for a, b in zip(pi1_costs, floor_costs) if a < b)
    mp, mf = sum(pi1_costs) / n, sum(floor_costs) / n
    return {"n": n, "pi1_mean": mp, "floor_mean": mf, "wins": wins,
            "pass": bool(mp <= mf)}


# --------------------------------------------------------------------------- #
# performance gates G1 / G4 / G5 / G7 (copied semantics of scripts/wave7_analysis.py)
# --------------------------------------------------------------------------- #
def sign_p(wins: int, n: int) -> float:
    """One-sided exact binomial P(X >= wins), H0 p = 0.5."""
    if n <= 0:
        return float("nan")
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def paired(trained: dict, ref: dict) -> dict:
    idx = sorted(set(trained) & set(ref))
    deltas = [trained[i] - ref[i] for i in idx]
    rel = [(trained[i] - ref[i]) / ref[i] for i in idx if ref[i]]
    wins = sum(1 for d in deltas if d < 0)
    n = len(deltas)
    return {"n": n, "wins": wins,
            "median_delta": median(deltas) if deltas else float("nan"),
            "median_rel": median(rel) if rel else float("nan"),
            "median_trained": median([trained[i] for i in idx]),
            "median_untrained": median([ref[i] for i in idx]),
            "sign_p": sign_p(wins, n)}


def g1_seed(trained: dict, untrained: dict) -> dict:
    """Wave-7 C1 verbatim: median reduction >= 15 % in BOTH variants (ratio of medians and
    median of per-path reductions) AND >= 20/30 paired wins (ceil(2n/3) below 30 pairs)
    AND one-sided sign-test p < 0.02."""
    pa = paired(trained, untrained)
    red1 = 1 - pa["median_trained"] / pa["median_untrained"] if pa["median_untrained"] else float("nan")
    red2 = -pa["median_rel"]
    need = max(1, math.ceil(0.667 * pa["n"])) if pa["n"] < 30 else 20
    ok = red1 >= 0.15 and red2 >= 0.15 and pa["wins"] >= need and pa["sign_p"] < 0.02
    return {**pa, "reduction_of_medians": red1, "median_of_reductions": red2,
            "wins_needed": need, "pass": bool(pa["n"] and ok)}


def g1_arm(seed_results: list[dict]) -> dict:
    """An arm passes G1 iff >= ceil(2/3 of its seeds) pass (3 seeds -> 2; 2 seeds -> both)."""
    n = len(seed_results)
    need = math.ceil(2 * n / 3) if n else 0
    passes = sum(1 for r in seed_results if r["pass"])
    return {"seeds": n, "passes": passes, "needed": need, "pass": bool(n and passes >= need)}


def cost_stats(costs: list[float]) -> dict:
    n = len(costs)
    if not n:
        return {"n": 0}
    m = sum(costs) / n
    sd = statistics.stdev(costs) if n >= 2 else 0.0
    return {"n": n, "mean": m, "median": median(costs), "max": max(costs), "std": sd,
            "cv": sd / m if m else float("nan"), "se": sd / math.sqrt(n) if n >= 2 else 0.0}


def g4_triplet(trained: list[float], untrained: list[float]) -> dict:
    """Paper-direction triplet on the sampled step arm: mean down >= 20 %, CV down >= 25 %,
    max down (the untrained sampled readout: median 22,207; CV 0.48 in wave 7)."""
    t, u = cost_stats(trained), cost_stats(untrained)
    mean_red = 1 - t["mean"] / u["mean"]
    cv_red = 1 - t["cv"] / u["cv"] if u["cv"] else float("nan")
    return {"trained": t, "untrained": u, "mean_reduction": mean_red, "cv_reduction": cv_red,
            "max_down": bool(t["max"] < u["max"]),
            "pass": bool(mean_red >= 0.20 and cv_red >= 0.25 and t["max"] < u["max"])}


def g5_agreement(per_seed_costs: list[list[float]]) -> dict:
    """Every seed pair's mean gap within 2*sqrt(SE_i^2 + SE_j^2). A zero-variance seed
    (a constant policy: SE = 0) that disagrees is 'mechanically disagreeing' — flagged,
    not scored as agreement (wave-7 RLCD z up to 132 was exactly that)."""
    cs = [cost_stats(c) for c in per_seed_costs]
    agree, mech = True, False
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            comb = 2 * math.sqrt(cs[i]["se"] ** 2 + cs[j]["se"] ** 2)
            if abs(cs[i]["mean"] - cs[j]["mean"]) > max(comb, 1e-9):
                agree = False
                if cs[i]["std"] == 0 or cs[j]["std"] == 0:
                    mech = True
    return {"agree": agree, "mechanically_disagreeing": mech,
            "zero_variance_seeds": [i for i, c in enumerate(cs) if c.get("std") == 0],
            "per_seed_mean": [c["mean"] for c in cs]}


def g7_parse(parse_fail: int, min_mass: float | None, trained: bool) -> dict:
    """parse_fail == 0 always; min label mass recorded and >= 0.9 for trained adapters."""
    mass_ok = True if (not trained or min_mass is None) else min_mass >= 0.9
    return {"parse_fail": parse_fail, "min_mass": min_mass,
            "pass": bool(parse_fail == 0 and mass_ok),
            "flag": None if mass_ok else "min_mass < 0.9: readout leaves the label set"}


# --------------------------------------------------------------------------- #
# calibration (report-only, W8 §6.4)
# --------------------------------------------------------------------------- #
def ece(conf: list[float], correct: list[int], bins: int = 10) -> float:
    n = len(conf)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i in range(n) if (lo <= conf[i] < hi) or (b == bins - 1 and conf[i] == 1.0)]
        if sel:
            tot += len(sel) / n * abs(sum(correct[i] for i in sel) / len(sel)
                                      - sum(conf[i] for i in sel) / len(sel))
    return tot


def brier(conf: list[float], correct: list[int]) -> float:
    return sum((c - y) ** 2 for c, y in zip(conf, correct)) / len(conf)


def auroc(conf: list[float], correct: list[int]) -> float:
    """P(conf of a positive > conf of a negative), ties 0.5; NaN with a single class."""
    pos = [c for c, y in zip(conf, correct) if y]
    neg = [c for c, y in zip(conf, correct) if not y]
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def calibration_table(P: list[list[float]], J: list[list[float]], tol: float = 0.01) -> dict:
    """confidence = max pi vs the indicator 'the chosen candidate's J is within `tol` (1 %)
    of the row minimum'; plus mean NLL of the softmax-minus-J target is left to the caller."""
    conf = [max(p) for p in P]
    correct = [1 if j[argmax_index(p)] <= min(j) * (1 + tol) else 0 for p, j in zip(P, J)]
    return {"n": len(P), "ece": ece(conf, correct), "brier": brier(conf, correct),
            "auroc": auroc(conf, correct), "accuracy": sum(correct) / len(correct),
            "mean_confidence": sum(conf) / len(conf)}


def select_lr(val_costs_by_lr: dict[str, dict[int, list[float]]], default: str) -> dict:
    """Pooled mean validation cost over all seeds per lr; lower wins; ties -> `default`
    (the comparability twin). val_costs_by_lr[lr_tag][seed] = per-run costs."""
    pooled = {}
    for lr, by_seed in val_costs_by_lr.items():
        xs = [c for costs in by_seed.values() for c in costs]
        pooled[lr] = sum(xs) / len(xs) if xs else float("inf")
    best = min(pooled.values())
    winners = [k for k, v in pooled.items() if v == best]
    return {"pooled_mean_val_cost": pooled,
            "winner": default if default in winners else sorted(winners)[0]}


# --------------------------------------------------------------------------- #
# small I/O helpers shared by the wave-8 scripts
# --------------------------------------------------------------------------- #
def update_json_key(path, key: str, value, *, force: bool = True) -> None:
    """Atomically set `key` in a json object file (created if missing). With
    force=False an existing non-null value is refused (F3 is frozen before training)."""
    import json
    import os
    from pathlib import Path
    p = Path(path)
    cur = json.loads(p.read_text()) if p.exists() else {}
    if not force and cur.get(key) is not None:
        raise SystemExit(f"{p}: {key} already set — frozen before training; pass --force to overwrite")
    cur[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p)
