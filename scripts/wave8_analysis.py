#!/usr/bin/env python3
"""wave8_analysis.py — WAVE8 gate readout (stdlib only; docs/waves/WAVE8_SPEC.md §6, §7.1).

Reads
  results/eval/w8-<arm>-seed<S>-<mode>-<pattern>.jsonl   protocol arms (decision_eval schema)
  results/eval/w8-untrained[-<rung>]-<mode>-<pattern>.jsonl  the untrained control(s) (C0)
  results/eval/w8-<arm>-seed<S>-perm.json                permutation gate probes
  results/wave8/<arm>-seed<S>-S_nom.json                 collapse-final reads (+ calibration)
  results/wave8/gates.json                               pre-flight F0-F3 (+ rung files gates_<rung>.json)
  results/wave7_floors.json                              30-pass floors (policy-independent)
  results/local-eval/trained-{step,noisy}/...            wave-6 a0_base trained arm (G3 reference)
Arm names: r1 | r2sft | r2rl | r3a | r3b | r3c, with a rung suffix -1.7b / -4b for the ladder
(control = untrained / untrained-1.7b / untrained-4b).

  python3 scripts/wave8_analysis.py [--out results/wave8_report.md] [--json results/wave8_report.json]
  python3 scripts/wave8_analysis.py select-lr  --val-dir results/wave8_val --out results/wave8/lr_selection.json
  python3 scripts/wave8_analysis.py select-sft-lr --results results --out results/wave8/sft_lr_selection.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_bullwhip import decision_gates as G                                  # noqa: E402

MODES = ("argmax", "sampled")
PATTERNS = ("step", "noisy")
RUNGS = ("0.6b", "1.7b", "4b")
ROGUE = 1e5
LR_DEFAULT = "lr1e-5"


# ------------------------------------------------------------------ loading ----
def parse_name(stem: str):
    """w8-<arm>[-seed<S>]-<mode>-<pattern> -> (arm, seed|None, mode, pattern)."""
    if not stem.startswith("w8-"):
        return None
    parts = stem[3:].split("-")
    if len(parts) < 3 or parts[-1] not in PATTERNS or parts[-2] not in MODES:
        return None
    rest = "-".join(parts[:-2])
    m = re.fullmatch(r"(.+)-seed(\d+)", rest)
    return (m.group(1), int(m.group(2)), parts[-2], parts[-1]) if m else (rest, None, parts[-2], parts[-1])


def load_arm(path: Path) -> dict:
    complete, failed = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (failed if (r.get("total_cost") is None or not r.get("orders")) else complete).append(r)
    return {"complete": complete, "failed": failed, "path": str(path)}


def load_eval_dir(d: Path) -> dict:
    arms = {}
    for p in sorted(d.glob("w8-*.jsonl")):
        k = parse_name(p.stem)
        if k:
            arms[k] = load_arm(p)
    return arms


def run_costs(arm: dict) -> dict:
    return {int(r["run"]): float(r["total_cost"]) for r in arm["complete"]}


def max_order(rec: dict) -> float:
    o = [float(x) for v in (rec.get("orders") or {}).values() if isinstance(v, list) for x in v]
    return max(o) if o else 0.0


def control_name(family: str) -> str:
    for rung in ("1.7b", "4b"):
        if family.endswith("-" + rung):
            return "untrained-" + rung
    return "untrained"


def rung_of(family: str) -> str:
    for rung in ("1.7b", "4b"):
        if family.endswith("-" + rung):
            return rung
    return "0.6b"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def floors_noisy(path: Path) -> dict:
    fl = load_json(path) or []
    out = {}
    for key, col in (("wf_tuned", "wf_cost"), ("untuned", "untuned_cost"), ("oracle", "oracle_cost")):
        xs = [r[col] for r in fl if r["pattern"] == "noisy" and r["pass"] >= 0]
        out[key] = statistics.median(xs) if xs else None
    return out


# -------------------------------------------------------------- analysis ----
def analyse(eval_dir: Path, wave8: Path, floors: Path, c3_step: Path, c3_noisy: Path) -> dict:
    arms = load_eval_dir(eval_dir)
    gates = load_json(wave8 / "gates.json") or {}
    out: dict = {"preflight": {}, "families": {}, "g2": {}, "g3": {}, "decision": {}}
    pf = out["preflight"]
    pf["F0"] = ((gates.get("F0") or {}).get("verdict") or {}).get("pass")
    pf["F1"] = ((gates.get("F1") or {}).get("verdict") or {}).get("pass")
    f2 = gates.get("F2")
    pf["F2"] = bool(f2 and f2.get("contiguous") and f2.get("prompt_sha_equal")) if f2 else None
    pf["F3"] = bool(gates.get("F3")) if "F3" in gates else None
    pf["clean"] = all(v is True for v in pf.values()) if pf else False

    fl = floors_noisy(floors)
    out["floors_noisy_median"] = fl
    wf = fl.get("wf_tuned")

    fams = sorted({a for (a, s, m, p) in arms if a.split("-")[0] != "untrained" and s is not None})
    for fam in fams:
        ctrl_name = control_name(fam)
        seeds = sorted({s for (a, s, m, p) in arms if a == fam and s is not None})
        fam_res: dict = {"control": ctrl_name, "rung": rung_of(fam), "seeds": {}}
        g1s, g4s = [], []
        for s in seeds:
            sres: dict = {}
            tag = f"{fam}-seed{s}"
            tr_n = arms.get((fam, s, "argmax", "noisy"))
            ct_n = arms.get((ctrl_name, None, "argmax", "noisy"))
            if tr_n and ct_n:
                sres["g1"] = G.g1_seed(run_costs(tr_n), run_costs(ct_n))
                g1s.append(sres["g1"])
                if wf:
                    med = sres["g1"]["median_trained"]
                    sres["g2"] = {"median": med, "x_wf_tuned": med / wf,
                                  "tier": ("floor-parity" if med <= 1.25 * wf else
                                           "headline" if med <= 2 * wf else
                                           "strong" if med <= 4 * wf else "over")}
            tr_s = arms.get((fam, s, "sampled", "step"))
            ct_s = arms.get((ctrl_name, None, "sampled", "step"))
            if tr_s and ct_s:
                sres["g4"] = G.g4_triplet(list(run_costs(tr_s).values()), list(run_costs(ct_s).values()))
                g4s.append(sres["g4"])
            # G7: parse + mass from the summaries
            summ = load_json(eval_dir / f"w8-{tag}-argmax-noisy.summary.json")
            if summ:
                sres["g7"] = G.g7_parse(int(summ.get("parse_fail", 0)), summ.get("min_mass"), True)
            # G6: collapse + permutation
            col = load_json(wave8 / f"{tag}-S_nom.json")
            perm = load_json(eval_dir / f"w8-{tag}-perm.json")
            stopped = (Path(eval_dir).parent / f"wave8_{fam.split('-')[0]}_{'*'}").exists() if False else None
            sres["g6"] = {"collapsed": (col or {}).get("collapse", {}).get("collapsed"),
                          "collapse_read": bool(col), "perm_rate": (perm or {}).get("invariance_rate"),
                          "perm_pass": (perm or {}).get("pass") if perm and perm.get("gate", 0) else None,
                          "perm_read": bool(perm),
                          "calibration": (col or {}).get("calibration")}
            fam_res["seeds"][s] = sres
        fam_res["g1_arm"] = G.g1_arm(g1s) if g1s else None
        costs = [list(run_costs(arms[(fam, s, "argmax", "noisy")]).values()) for s in seeds
                 if (fam, s, "argmax", "noisy") in arms]
        fam_res["g5"] = G.g5_agreement(costs) if len(costs) >= 2 else None
        fam_res["g4_pass_seeds"] = sum(1 for g in g4s if g["pass"])
        fam_res["n_seeds"] = len(seeds)
        out["families"][fam] = fam_res

    # G3 vs wave-6 a0_base (identical paths)
    for pat, ref_path, arm_pat, mode, first in (("step", c3_step, "step", "sampled", None),
                                                 ("noisy", c3_noisy, "noisy", "argmax", 10)):
        if not ref_path.exists():
            out["g3"][pat] = "reference missing"
            continue
        ref = load_arm(ref_path)
        ref_costs = run_costs(ref)
        rogue = {int(r["run"]) for r in ref["complete"] if max_order(r) >= ROGUE}
        for fam in fams:
            for s in sorted({s for (a, s, m, p) in arms if a == fam}):
                a = arms.get((fam, s, mode, arm_pat))
                if not a:
                    continue
                t = run_costs(a)
                if first:
                    t = {i: v for i, v in t.items() if i < first}
                raw = G.paired(t, ref_costs)
                clean_t = {i: v for i, v in t.items() if i not in rogue}
                clean = G.paired(clean_t, {i: ref_costs[i] for i in clean_t if i in ref_costs})
                out["g3"][f"{fam}-seed{s}|{pat}"] = {"raw": raw, "clean": clean,
                                                      "ref_rogue_runs": sorted(rogue)}

    # decision table (§0.1 / §6.6)
    reasons, verdict = [], "INCONCLUSIVE"
    r1_rungs_failed, r1_rungs_seen, r2_seen, r2_failed = set(), set(), False, False
    supporting = False
    for fam, fr in out["families"].items():
        g1 = fr["g1_arm"]
        if not g1:
            continue
        base = fam.split("-")[0]
        if base == "r1":
            r1_rungs_seen.add(fr["rung"])
            if not g1["pass"]:
                r1_rungs_failed.add(fr["rung"])
        if base in ("r2sft", "r2rl"):
            r2_seen = True
            r2_failed = r2_failed or not g1["pass"]
        if base in ("r1", "r2sft", "r2rl") and g1["pass"]:
            seeds = fr["seeds"]
            collapsed = any((v.get("g6") or {}).get("collapsed") for v in seeds.values())
            if collapsed:
                reasons.append(f"{fam}: passes G1 but a seed is collapsed — cannot satisfy SUPPORTING")
            elif fr["g4_pass_seeds"] < g1["needed"]:
                reasons.append(f"{fam}: passes G1 but the G4 paper-direction triplet holds on "
                               f"{fr['g4_pass_seeds']}/{fr['n_seeds']} seeds (< {g1['needed']})")
            else:
                supporting = True
                reasons.append(f"{fam}: G1 {g1['passes']}/{g1['seeds']} seeds, G4 triplet, not collapsed")
    if not pf["clean"]:
        reasons.append(f"pre-flight not clean: {pf}")
    if supporting and pf["clean"]:
        verdict = "SUPPORTING (bounded, mechanism-level; see §0.1 for allowed wording)"
    elif (pf["clean"] and set(RUNGS) <= r1_rungs_seen and r1_rungs_failed == set(RUNGS)
          and r2_seen and r2_failed):
        verdict = "CONTRADICTING (bounded: outcome-scored RL at <= 4B with this recipe/budget; not the paper's autoregressive result)"
    else:
        if not reasons:
            reasons.append("no arm cleared G1+G4 with a clean collapse read, and not every rung failed")
    out["decision"] = {"verdict": verdict, "reasons": reasons,
                       "r1_rungs_seen": sorted(r1_rungs_seen), "r1_rungs_failed": sorted(r1_rungs_failed)}
    return out


def select_lr_from_val(val_dir: Path, default: str = LR_DEFAULT, family: str = "r1") -> dict:
    by_lr: dict = {}
    for p in sorted(val_dir.glob("w8-val-*.jsonl")):
        m = re.fullmatch(r"w8-val-(" + re.escape(family) + r")-(lr[0-9.]+e[-+]?[0-9]+)-seed(\d+)", p.stem)
        if not m:
            continue
        by_lr.setdefault(m.group(2), {})[int(m.group(3))] = list(run_costs(load_arm(p)).values())
    if not by_lr:
        raise SystemExit(f"no w8-val-*.jsonl under {val_dir}")
    res = G.select_lr(by_lr, default)
    res["seeds_per_lr"] = {k: sorted(v) for k, v in by_lr.items()}
    return res


def select_sft_lr(results: Path) -> dict:
    """lr from the 500-step SFT sweeps (seed 0) by validation NLL at the last val event.

    The sweep set is discovered on disk (results/wave8_r2sft_lrsweep_<tag>/) rather than
    hardcoded: the arm swept {1e-5, 1e-4, 2e-4} and a hardcoded pair silently ignored any
    lr outside it, which would pick a winner from an incomplete comparison.
    """
    nll = {}
    for p in sorted(results.glob("wave8_r2sft_lrsweep_*/train_log.jsonl")):
        tag = p.parent.name.replace("wave8_r2sft_lrsweep_", "")
        vals = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        v = [r for r in vals if r.get("event") == "val"]
        if not v:
            continue
        last = max(v, key=lambda r: r.get("step", -1))          # val_every=100, steps=500 -> step 499
        nll[tag] = last["nll"]
    if not nll:
        raise SystemExit("no validation events in results/wave8_r2sft_lrsweep_*/train_log.jsonl "
                         "(sweep not finished, or train_log.jsonl missing)")
    best = min(nll.values())
    winners = [k for k, x in nll.items() if x == best]
    return {"val_nll_last": nll, "n_candidates": len(nll),
            "winner": LR_DEFAULT if LR_DEFAULT in winners else sorted(winners)[0]}


# -------------------------------------------------------------- rendering ----
def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else (f"{x:,.{nd}f}" if isinstance(x, float) else str(x))


def render(res: dict) -> str:
    L = ["# WAVE8 gate readout", ""]
    L.append(f"pre-flight: {res['preflight']}")
    L.append(f"noisy floor medians (30 passes): {res['floors_noisy_median']}")
    L.append("")
    L.append("## G1 (noisy-argmax n=30 vs the arm's own untrained readout) + G2 + G4 + G6 + G7, per seed")
    L.append("")
    L.append("| arm | seed | G1 red(med) | red(path-med) | wins | sign p | G1 | G2 x wf | tier | G4 | collapsed | perm | G7 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam, fr in res["families"].items():
        for s, r in fr["seeds"].items():
            g1, g2, g4, g6, g7 = (r.get(k) or {} for k in ("g1", "g2", "g4", "g6", "g7"))
            L.append(f"| {fam} | {s} | {f(g1.get('reduction_of_medians'), 3)} | {f(g1.get('median_of_reductions'), 3)} | "
                     f"{g1.get('wins', 'n/a')}/{g1.get('n', 'n/a')} | {f(g1.get('sign_p'), 4)} | "
                     f"{('PASS' if g1.get('pass') else 'fail') if g1 else 'n/a'} | {f(g2.get('x_wf_tuned'), 2)} | "
                     f"{g2.get('tier', 'n/a')} | {('PASS' if g4.get('pass') else 'fail') if g4 else 'n/a'} | "
                     f"{f(g6.get('collapsed'))} | {f(g6.get('perm_rate'))} | "
                     f"{('ok' if g7.get('pass') else 'FLAG') if g7 else 'n/a'} |")
    L.append("")
    L.append("## Arm level")
    for fam, fr in res["families"].items():
        g5 = fr["g5"]
        L.append(f"- **{fam}** (control {fr['control']}, rung {fr['rung']}): G1 arm {fr['g1_arm']}; "
                 f"G4 seeds passing {fr['g4_pass_seeds']}/{fr['n_seeds']}; G5 {g5}")
    L.append("")
    L.append("## G3 vs wave-6 a0_base (paired on identical paths; raw / rogue-clean)")
    for k, v in res["g3"].items():
        if isinstance(v, str):
            L.append(f"- {k}: {v}")
        else:
            L.append(f"- {k}: raw wins {v['raw']['wins']}/{v['raw']['n']} (sign p {f(v['raw']['sign_p'], 4)}), "
                     f"clean wins {v['clean']['wins']}/{v['clean']['n']}")
    L.append("")
    d = res["decision"]
    L.append(f"## Decision (§0.1 / §6.6): **{d['verdict']}**")
    for r in d["reasons"]:
        L.append(f"- {r}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    rp = sub.add_parser("report")
    sl = sub.add_parser("select-lr")
    sl.add_argument("--val-dir", default="results/wave8_val")
    sl.add_argument("--family", default="r1", help="r1 | r1-1.7b | r1-4b | r2rl")
    sl.add_argument("--out", default="results/wave8/lr_selection.json")
    ss = sub.add_parser("select-sft-lr")
    ss.add_argument("--results", default="results")
    ss.add_argument("--out", default="results/wave8/sft_lr_selection.json")
    for p in (ap, rp):
        p.add_argument("--eval-dir", default="results/eval")
        p.add_argument("--wave8", default="results/wave8")
        p.add_argument("--floors", default="results/wave7_floors.json")
        p.add_argument("--c3-step", default="results/local-eval/trained-step/qwen3-0.6b-trained-baseline.jsonl")
        p.add_argument("--c3-noisy", default="results/local-eval/trained-noisy/qwen3-0.6b-trained-baseline.jsonl")
        p.add_argument("--out", default=None)
        p.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "select-lr":
        res = select_lr_from_val(Path(a.val_dir), family=a.family)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"lr selection: winner {res['winner']} pooled means {res['pooled_mean_val_cost']}")
        return 0
    if a.cmd == "select-sft-lr":
        res = select_sft_lr(Path(a.results))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"SFT lr selection: winner {res['winner']} val nll {res['val_nll_step499']}")
        return 0
    res = analyse(Path(a.eval_dir), Path(a.wave8), Path(a.floors), Path(a.c3_step), Path(a.c3_noisy))
    text = render(res)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
