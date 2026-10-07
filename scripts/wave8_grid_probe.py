#!/usr/bin/env python3
"""wave8_grid_probe.py — WAVE8 pre-flight F0 (docs/waves/WAVE8_SPEC.md §6.1, Appendix A productionised).

For N states per behaviour regime (mixed = the wave-7 round-1 generator; nominal = floor
theta=3.5, lam=0.2 only) score E[J] over the REFERENCE grid 0..200 step 2 (101 candidates,
M CRN futures) ONCE per state, then read the choice-set quality of w8_g26 and of the wave-7
grid w7_u21 from the same J vector: relative regret, target mass / argmax share at the top
letter. Verdict from decision_gates.f0_gate (thresholds fixed in the spec, not here).

  python3 scripts/wave8_grid_probe.py --states 1000 --processes 6 \
      --out results/wave8/f0.json --gates results/wave8/gates.json
  # independent re-run for VERIFY: --state-seed 3 --label-seed 6050 --out /tmp/f0_check.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_bullwhip import decision_labeler as L                                # noqa: E402
from agent_bullwhip.decision_gates import f0_gate, f0_stats, update_json_key    # noqa: E402
from agent_bullwhip.decision_readout import get_grid                             # noqa: E402
from agent_bullwhip.engine import ROLES                                          # noqa: E402

REGIMES = {"mixed": ((0.75, 0.25), None), "nominal": ((1.0, 0.0), ((3.5, 3.5), (0.2, 0.2)))}
CURR = ("poisson", "truncnorm")


def _task(t: dict) -> dict:
    """One state: replay the behaviour episode to `week`, score the reference grid."""
    (pf, pr), ranges = REGIMES[t["regime"]]
    beh = L.BehaviorConfig(pf, pr, *ranges) if ranges else L.BehaviorConfig(pf, pr)
    seed, ep, week, role, cur = t["label_seed"], t["ep"], t["week"], t["role"], t["cur"]
    step, il, params = L._replay_to_week(seed, ep, week, cur, 36, beh)
    ctx = step.observations()[role]
    cont = L.make_continuation(L.CONTINUATION_THETA, L.CONTINUATION_LAM, il, week)
    rng = L._seeded_rng(f"{seed}:ep:{ep}:w:{week}:{role}")
    fut = [L.draw_future_demand(rng, cur, params, 36 - week) for _ in range(t["m"])]
    J = L.score_candidates(step, week, role, L.REFERENCE_GRID, cont, fut,
                           L.CONTINUATION_THETA, L.CONTINUATION_LAM, il)
    return {"role": role, "week": week, "ep": ep, "J": J,
            "ctx": {k: ctx[k] for k in ("on_hand", "backlog", "outstanding", "incoming_last")}}


def build_tasks(regime: str, n: int, state_seed: int, label_seed: int, m: int) -> list[dict]:
    rnd = random.Random(f"{regime}:{state_seed}")
    out = []
    for _ in range(n):
        ep = rnd.randrange(700)
        out.append({"regime": regime, "ep": ep, "week": rnd.randrange(34),
                    "role": rnd.choice(ROLES), "cur": CURR[ep % 2],
                    "label_seed": label_seed, "m": m})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=1000, help="states per regime (spec: >= 1000)")
    ap.add_argument("--regimes", default="mixed,nominal")
    ap.add_argument("--m-futures", type=int, default=8)
    ap.add_argument("--state-seed", type=int, default=1)
    ap.add_argument("--label-seed", type=int, default=6000)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--grid", default="w8_g26b")
    ap.add_argument("--out", default="results/wave8/f0.json")
    ap.add_argument("--gates", default=None, help="merge the verdict into this gates.json (key F0)")
    a = ap.parse_args(argv)
    if L.EVAL_SEED_MIN <= a.label_seed < L.EVAL_SEED_MAX:
        raise SystemExit("label seed is in the reserved eval band")
    grid, w7 = list(get_grid(a.grid).values), list(get_grid("w7_u21").values)
    t0 = time.time()
    stats, w7stats, rows_out = {}, {}, {}
    for regime in a.regimes.split(","):
        tasks = build_tasks(regime, a.states, a.state_seed, a.label_seed, a.m_futures)
        if a.processes > 1:
            with mp.get_context("fork").Pool(a.processes) as pool:
                rows = list(pool.imap(_task, tasks, chunksize=4))
        else:
            rows = [_task(t) for t in tasks]
        J = [r["J"] for r in rows]
        stats[regime] = f0_stats(J, grid, L.REFERENCE_GRID)
        w7stats[regime] = f0_stats(J, w7, L.REFERENCE_GRID)
        rows_out[regime] = [{k: r[k] for k in ("role", "week", "ep", "ctx")} | {"J": [round(x, 3) for x in r["J"]]}
                            for r in rows]
        s = stats[regime]
        print(f"[{regime}] n={s['n_states']} regret mean={100*s['regret_mean_rel']:.3f}% "
              f"p90={100*s['regret_p90_rel']:.3f}% top_mass={s['top_mass']:.4f} "
              f"top_argmax={s['top_argmax_share']:.4f} | wave-7 grid regret "
              f"{100*w7stats[regime]['regret_mean_rel']:.2f}% top_mass={w7stats[regime]['top_mass']:.3f}",
              flush=True)
    verdict = f0_gate(stats) if set(stats) == {"mixed", "nominal"} else {"pass": None, "reason": "need both regimes"}
    res = {"grid": a.grid, "reference": [0, 200, 2], "m_futures": a.m_futures,
           "states_per_regime": a.states, "state_seed": a.state_seed, "label_seed": a.label_seed,
           "wall_sec": round(time.time() - t0, 1), "processes": a.processes,
           "stats": stats, "wave7_grid_stats": w7stats, "verdict": verdict}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    Path(str(out) + ".J.json").write_text(json.dumps(rows_out))
    if a.gates:
        update_json_key(a.gates, "F0", {**{k: res[k] for k in ("grid", "m_futures", "states_per_regime",
                                                              "state_seed", "label_seed")},
                                        "stats": stats, "verdict": verdict, "source": str(out)})
    print(f"F0 verdict: {'PASS' if verdict['pass'] else 'FAIL'}  wall={res['wall_sec']}s -> {out}")
    return 0 if verdict.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
