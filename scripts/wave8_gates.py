#!/usr/bin/env python3
"""wave8_gates.py — WAVE8 fixed state sample, frozen control stats, collapse + permutation gates
(docs/waves/WAVE8_SPEC.md §6.1-§6.4). Gate math lives in agent_bullwhip/decision_gates.py.

Subcommands
  sample          S_nom: 1,024 states = every 2nd state of 16 nominal-floor (theta 3.5, lam 0.2)
                  curriculum episodes (seed 9100+i, curriculum alternating), weeks < 34, 4 roles
                  [hub CPU]                    -> results/wave8/S_nom.jsonl
  oracle          simulator oracle labels on S_nom (labeler M=16, w8_g26) -> S_nom_oracle.json
                  [hub CPU, ~10 core-min]
  control         untrained readout on S_nom -> freeze F2 (interface) + F3 (collapse thresholds)
                  into the gates json  [needs torch; cluster pre-flight job]
  collapse-check  trained adapter on every 4th S_nom state (256) vs the frozen F3 -> --out json
  collapse-final  all 1,024 states + per-letter histogram + calibration -> results/wave8/<arm>-S_nom.json
  perm-check      permutation gate (delegates to probe_permutation_invariance.py logic)
  report          print the gate table from a gates json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_bullwhip import decision_gates as G                                   # noqa: E402
from agent_bullwhip import decision_labeler as L                                 # noqa: E402
from agent_bullwhip.decision_readout import (PINNED_REVISION, get_grid, prompt_hash,   # noqa: E402
                                             render_prompt, render_prompt_str)
from agent_bullwhip.engine import ROLES                                          # noqa: E402

S_NOM = "results/wave8/S_nom.jsonl"
NOMINAL = ((1.0, 0.0), ((3.5, 3.5), (0.2, 0.2)))
CURR = ("poisson", "truncnorm")
N_EPISODES, N_WEEKS, STRIDE, N_STATES = 16, 34, 2, 1024
SEED0 = 9100


# ------------------------------------------------------------------ sample ----
def build_s_nom() -> list[dict]:
    """16 x 34 x 4 = 2,176 states in (episode, week, role) order -> stride 2 -> first 1,024."""
    beh = L.BehaviorConfig(1.0, 0.0, (3.5, 3.5), (0.2, 0.2))
    rows = []
    for i in range(N_EPISODES):
        seed, cur = SEED0 + i, CURR[i % 2]
        for week in range(N_WEEKS):
            step, _, params = L._replay_to_week(seed, 0, week, cur, 36, beh)
            obs = step.observations()
            for role in ROLES:
                rows.append({"episode": i, "week": week, "role": role, "curriculum": cur,
                             "seed": seed, "params": params, "ctx": dict(obs[role])})
    return rows[::STRIDE][:N_STATES]


def cmd_sample(a) -> int:
    rows = build_s_nom()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    Path(a.out).write_text(raw)
    import hashlib
    sha = hashlib.sha256(raw.encode()).hexdigest()
    print(f"S_nom: {len(rows)} states, seeds {SEED0}..{SEED0 + N_EPISODES - 1}, sha256 {sha[:16]} -> {a.out}")
    if a.gates:
        G.update_json_key(a.gates, "S_nom", {"n": len(rows), "sha256": sha, "path": a.out,
                                            "seeds": [SEED0, SEED0 + N_EPISODES - 1]})
    return 0


def load_s_nom(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ oracle ----
def _oracle_task(t):
    return L._label_state_task(t)


def cmd_oracle(a) -> int:
    rows = load_s_nom(a.states)
    grid = a.grid
    tasks = [{"seed": r["seed"], "ep": 0, "week": r["week"], "role": r["role"],
              "curriculum": r["curriculum"], "m_futures": a.m_futures,
              "theta": L.CONTINUATION_THETA, "lam": L.CONTINUATION_LAM, "tau_scale": 0.5,
              "horizon": 36, "p_floor": 1.0, "p_random": 0.0, "theta_range": (3.5, 3.5),
              "lam_range": (0.2, 0.2), "grid_id": grid} for r in rows]
    if a.processes > 1:
        with mp.get_context("fork").Pool(a.processes) as pool:
            lab = list(pool.imap(_oracle_task, tasks, chunksize=4))
    else:
        lab = [_oracle_task(t) for t in tasks]
    for r, l in zip(rows, lab):
        if l["ctx"] != r["ctx"]:
            raise SystemExit("oracle replay disagrees with S_nom ctx — replay/seed mismatch")
    argmin = [l["argmin_idx"] for l in lab]
    counts = {}
    for x in argmin:
        counts[x] = counts.get(x, 0) + 1
    modal = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    out = {"grid": grid, "n": len(rows), "m_futures": a.m_futures, "argmin": argmin,
           "modal_letter": modal[0], "m_star": modal[1] / len(rows),
           "m_hi": G.wilson_upper(modal[1], len(rows)), "J": [l["J"] for l in lab],
           "p": [l["p"] for l in lab]}
    Path(a.out).write_text(json.dumps(out))
    print(f"oracle: modal letter idx {modal[0]} share m*={out['m_star']:.4f} (Wilson upper "
          f"{out['m_hi']:.4f}) over {len(rows)} states -> {a.out}")
    if a.gates:
        G.update_json_key(a.gates, "S_nom_oracle", {"modal_letter": modal[0], "m_star": out["m_star"],
                                                    "m_hi": out["m_hi"], "m_futures": a.m_futures,
                                                    "path": a.out})
    return 0


# --------------------------------------------------- torch-dependent commands ----
def readout_probs(model: str, revision: str | None, adapter: str | None, grid_id: str,
                  ctxs: list[dict], batch: int = 32,
                  untrained_require_mass: float | None = None):
    """(probs[n][k], masses, label_ids, prompt_sha) for the adapter (None = untrained).

    WAVE8-A1: for the 1.7B/4B rungs the untrained readout does NOT concentrate 99 % of
    its token mass on the 26 candidate letters (measured on S_nom, all 1,024 states,
    fp32 CPU: Qwen3-1.7B min mass 0.348, 222/1024 prompts below 0.99; bf16 was not the
    cause — it reads HIGHER there). Following the wave-7 A1 precedent (trained adapters
    switched to record-not-assert), the untrained assert for the taller rungs becomes a
    RECORDED min_mass in the F2/F3 stats; F0/F1 grid + teacher gates stay untouched and
    remain fatal. Pass untrained_require_mass to change/restore it."""
    from agent_bullwhip.decision_readout import ReadoutModel
    rm = ReadoutModel(model, adapter_dir=adapter, revision=revision, grid_id=grid_id)
    require = 0.99 if adapter else (untrained_require_mass
                                    if untrained_require_mass is not None else 0.0)
    P, masses = [], []
    for lo in range(0, len(ctxs), batch):
        prompts = [render_prompt(c, rm.tokenizer, grid_id=grid_id) for c in ctxs[lo:lo + batch]]
        for rec in rm.readout_batch(prompts, require_mass=require):
            P.append(rec["probs"])
            masses.append(rec["mass"])
    return P, masses, dict(rm.label_ids), rm.prompt_hash


CANON = {"role": "retailer", "t": 0, "on_hand": 0, "backlog": 0, "outstanding": 0,
         "incoming_last": 4, "incoming_now": 4, "last_orders": []}


def cmd_control(a) -> int:
    rows = load_s_nom(a.states)
    oracle = json.loads(Path(a.oracle).read_text())
    P, masses, ids, psha = readout_probs(a.model, a.revision or None, None, a.grid,
                                         [r["ctx"] for r in rows], a.batch)
    string_sha = prompt_hash(render_prompt_str(CANON, grid_id=a.grid))
    vals = list(ids.values())
    f2 = {"grid": a.grid, "label_ids": ids, "contiguous": vals == list(range(vals[0], vals[0] + len(vals))),
          "prompt_sha_apply_chat_template": psha, "prompt_sha_string_renderer": string_sha,
          "prompt_sha_equal": psha == string_sha, "min_mass_untrained": min(masses)}
    f3 = G.freeze_control_stats(P, oracle["argmin"])
    f3.update({"model": a.model, "revision": a.revision, "min_mass": min(masses), "grid": a.grid})
    if not (f2["contiguous"] and f2["prompt_sha_equal"]):
        raise SystemExit(f"F2 FAILED: {f2}")
    G.update_json_key(a.gates, "F2", f2, force=a.force)
    G.update_json_key(a.gates, "F3", f3, force=a.force)
    Path(a.out).write_text(json.dumps({"probs": P, "masses": masses}))
    print(f"F2 ok (ids {vals[0]}..{vals[-1]}, prompt sha equal). F3 frozen: H_lo={f3['H_lo']:.3f} "
          f"H_hi={f3['H_hi']:.3f} m_hi={f3['m_hi']:.3f} I_hi0={f3['I_hi0']:.4f} -> {a.gates}")
    return 0


def _frozen(path: str) -> dict:
    g = json.loads(Path(path).read_text())
    if not g.get("F3"):
        raise SystemExit(f"{path}: F3 not frozen yet — run `wave8_gates.py control` first")
    return g["F3"]


def cmd_collapse_check(a) -> int:
    rows = load_s_nom(a.states)[::4][:a.n]
    P, masses, _, _ = readout_probs(a.model, a.revision or None, a.adapter, a.grid,
                                    [r["ctx"] for r in rows], a.batch)
    res = G.collapse_gate(P, _frozen(a.frozen))
    res.update({"adapter": a.adapter, "min_mass": min(masses), "n_checked": len(rows)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"collapse-check {a.adapter}: collapsed={res['collapsed']} medH={res['median_entropy']:.3f} "
          f"modal_share={res['modal_argmax_share']:.3f} I={res['I']:.4f} {res['reasons']}")
    return 0


def cmd_collapse_final(a) -> int:
    rows = load_s_nom(a.states)
    P, masses, _, _ = readout_probs(a.model, a.revision or None, a.adapter, a.grid,
                                    [r["ctx"] for r in rows], a.batch)
    frozen = _frozen(a.frozen)
    res = G.collapse_gate(P, frozen)
    k = len(P[0])
    hist = [0] * k
    for p in P:
        hist[G.argmax_index(p)] += 1
    out = {"arm": a.arm, "adapter": a.adapter, "grid": a.grid, "n_states": len(P),
           "collapse": res, "argmax_histogram": hist, "min_mass": min(masses),
           "oracle_modal_share": frozen["m_star"],
           "top_letter_share": hist[-1] / len(P), "zero_letter_share": hist[0] / len(P),
           "entropy_median": res["median_entropy"]}
    if a.labels:
        from agent_bullwhip.decision_train import load_labels, train_val_split
        lrows, _ = load_labels(a.labels)
        val = [lrows[i] for i in train_val_split(lrows, 0.05)]
        Pv, _, _, _ = readout_probs(a.model, a.revision or None, a.adapter, a.grid,
                                    [r["ctx"] for r in val], a.batch)
        out["calibration"] = G.calibration_table(Pv, [r["J"] for r in val])
        out["calibration"]["target_nll"] = sum(
            -sum(t * __import__("math").log(p + 1e-9) for t, p in zip(r["p"], pv))
            for r, pv in zip(val, Pv)) / len(val)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"collapse-final {a.arm}: collapsed={res['collapsed']} argmax-histogram top3="
          f"{sorted(enumerate(hist), key=lambda kv: -kv[1])[:3]} -> {a.out}")
    return 0


def cmd_perm_check(a) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe_perm", Path(__file__).with_name("probe_permutation_invariance.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    argv = ["--model", a.model, "--cases", a.states, "--n", str(a.n), "--perms", "3",
            "--gate", str(a.gate), "--grid", a.grid, "--out", a.out]
    if a.adapter:
        argv += ["--adapter", a.adapter]
    if a.revision:
        argv += ["--revision", a.revision]
    sys.argv = ["probe_permutation_invariance.py"] + argv
    return m.main()


def cmd_report(a) -> int:
    g = json.loads(Path(a.gates).read_text())
    print(f"# WAVE8 gates — {a.gates}")
    f0 = (g.get("F0") or {}).get("verdict")
    print(f"F0 grid     : {'PASS' if f0 and f0.get('pass') else ('FAIL' if f0 else 'not run')}")
    for regime, rows in ((f0 or {}).get("regimes") or {}).items():
        for k, v in rows.items():
            print(f"   {regime:8s} {k:18s} {v['value']:.5f} <= {v['threshold']} : {'ok' if v['pass'] else 'FAIL'}")
    f1 = (g.get("F1") or {}).get("verdict")
    print("F1 teacher  : " + (f"{'PASS' if f1['pass'] else 'FAIL'} (PI1 mean {f1['pi1_mean']:,.0f} vs floor "
                              f"{f1['floor_mean']:,.0f}; wins {f1['wins']}/{f1['n']})" if f1 else "not run"))
    f2 = g.get("F2")
    print("F2 interface: " + (f"{'PASS' if f2['contiguous'] and f2['prompt_sha_equal'] else 'FAIL'} "
                              f"(ids contiguous={f2['contiguous']}, prompt sha equal={f2['prompt_sha_equal']})"
                              if f2 else "not run (cluster pre-flight)"))
    f3 = g.get("F3")
    print("F3 control  : " + (f"frozen H_lo={f3['H_lo']:.3f} H_hi={f3['H_hi']:.3f} m_hi={f3['m_hi']:.3f} "
                              f"I_hi0={f3['I_hi0']:.4f} (m*={f3['m_star']:.3f})" if f3 else
                              "not frozen (cluster pre-flight)"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, torch=False):
        p.add_argument("--states", default=S_NOM)
        p.add_argument("--gates", default=None)
        p.add_argument("--grid", default="w8_g26b")
        if torch:
            p.add_argument("--model", default="Qwen/Qwen3-0.6B")
            p.add_argument("--revision", default=PINNED_REVISION)
            p.add_argument("--batch", type=int, default=32)
        return p
    s = common(sub.add_parser("sample")); s.add_argument("--out", default=S_NOM)
    o = common(sub.add_parser("oracle"))
    o.add_argument("--out", default="results/wave8/S_nom_oracle.json")
    o.add_argument("--m-futures", type=int, default=16)
    o.add_argument("--processes", type=int, default=1)
    c = common(sub.add_parser("control"), True)
    c.add_argument("--oracle", default="results/wave8/S_nom_oracle.json")
    c.add_argument("--out", default="results/wave8/S_nom_control.json")
    c.set_defaults(gates="results/wave8/gates.json")
    c.add_argument("--force", action="store_true")
    k = common(sub.add_parser("collapse-check"), True)
    k.add_argument("--adapter", default=None); k.add_argument("--n", type=int, default=256)
    k.add_argument("--frozen", default="results/wave8/gates.json"); k.add_argument("--out", required=True)
    f = common(sub.add_parser("collapse-final"), True)
    f.add_argument("--arm", required=True); f.add_argument("--adapter", default=None)
    f.add_argument("--frozen", default="results/wave8/gates.json"); f.add_argument("--out", required=True)
    f.add_argument("--labels", default=None)
    pm = common(sub.add_parser("perm-check"), True)
    pm.add_argument("--adapter", default=None); pm.add_argument("--n", type=int, default=512)
    pm.add_argument("--gate", type=float, default=0.90); pm.add_argument("--out", required=True)
    r = sub.add_parser("report"); r.add_argument("--gates", default="results/wave8/gates.json")
    a = ap.parse_args(argv)
    return {"sample": cmd_sample, "oracle": cmd_oracle, "control": cmd_control,
            "collapse-check": cmd_collapse_check, "collapse-final": cmd_collapse_final,
            "perm-check": cmd_perm_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
