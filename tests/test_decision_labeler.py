"""Wave 7 P2: decision_labeler — curricula parity, warm replay, branch J, targets.

Torch-free (the labeler is pure simulation): every test runs on the hub base
Python. The hand-computed J test pins the engine arithmetic with a continuation
that keeps all quantities integer-exact (theta=1, lam=0 -> q = max(0, -IP)).
"""
from __future__ import annotations

import json

import pytest

from agent_bullwhip.engine import ROLES, GameStepper, SimConfig, run_game
from agent_bullwhip.agents import OrderUpToAgent
from agent_bullwhip.grpo_env import sample_curriculum_demand
from agent_bullwhip.decision_labeler import (
    EVAL_SEED_MAX, EVAL_SEED_MIN, BehaviorConfig, _seeded_rng, branch_cost,
    draw_future_demand, enumerate_states, make_continuation,
    sample_curriculum_demand_ex, run_labeler, run_labeler_from_states,
    score_candidates,
    softmax_minus_j, warm_forecast,
)
from agent_bullwhip.decision_readout import ORDER_GRID


# --------------------------------------------------------------------------- #
# Curriculum parity (single source of truth stays in wave-6 grpo_env)
# --------------------------------------------------------------------------- #
class TestCurriculumParity:
    @pytest.mark.parametrize("curriculum", ["poisson", "truncnorm"])
    def test_call_identical_to_grpo_env(self, curriculum):
        """Same seeded rng -> identical demand AND identical residual rng state:
        the labeler's sampler consumes random() exactly as grpo_env's, so label
        demand is provably the wave-6 curriculum."""
        rng = __import__("random").Random(7)
        d1 = sample_curriculum_demand(rng, curriculum, 36)
        state1 = rng.getstate()
        rng2 = __import__("random").Random(7)
        params, d2 = sample_curriculum_demand_ex(rng2, curriculum, 36)
        assert d1 == d2
        assert rng2.getstate() == state1
        assert set(params) in ({"lam"}, {"mu", "sigma"})
        if curriculum == "truncnorm":
            assert 8 <= params["mu"] <= 20 and 2 <= params["sigma"] <= 6
        else:
            assert 5 <= params["lam"] <= 20

    def test_futures_drawn_from_episode_params(self):
        """draw_future_demand uses the episode's OWN params (no fresh U(5,20))."""
        import random
        rng = random.Random(3)
        p, d = sample_curriculum_demand_ex(rng, "poisson", 10)
        f1 = draw_future_demand(rng, "poisson", p, 5)
        f2 = draw_future_demand(rng, "poisson", p, 5)
        assert len(f1) == len(f2) == 5
        # bounded replica of the curriculum's value clamp
        assert all(0 <= v <= 50 for v in f1 + f2)


# --------------------------------------------------------------------------- #
# Warm replay (AUDIT-1 fix 2) and branch-vs-run_game equality
# --------------------------------------------------------------------------- #
class TestContinuation:
    def test_warm_replay_equals_warm_run(self):
        """A floor that played the game has the same forecast as replaying its
        observed incoming_last sequence from week 0 (pure-floor run)."""
        import random
        theta, lam = 3.0, 0.5
        rng = random.Random(11)
        demand = [4, 4, 8, 8, 8, 8]
        agents = {r: OrderUpToAgent(theta, lam) for r in ROLES}
        step = GameStepper(demand, SimConfig(horizon=len(demand)))
        il_hist: dict[str, list[int]] = {r: [] for r in ROLES}
        while not step.done():
            obs = step.observations()
            for r in ROLES:
                il_hist[r].append(obs[r]["incoming_last"])
            step.advance({r: agents[r](obs[r]) for r in ROLES})
        for r in ROLES:
            # replay exactly the weeks the live agent played (0..H-1)
            assert agents[r]._forecast == pytest.approx(
                warm_forecast(theta, lam, il_hist[r]), abs=1e-9)

    def test_make_continuation_warm_is_pre_update_state(self):
        """make_continuation(week=w) holds the PRE-update forecast of week w
        (roles called at week w update with that week's il themselves); the
        candidate role's POST-update warm (candidate_warm in branch_cost) is
        the live floor state after ALSO folding week w's il."""
        theta, lam = 3.0, 0.5
        il = {"retailer": [0, 4, 8], "wholesaler": [0], "distributor": [0],
              "factory": [0]}
        cont = make_continuation(theta, lam, il, week=2)
        live_pre = OrderUpToAgent(theta, lam)
        for v in il["retailer"][:2]:         # weeks 0..1 -> pre update of week 2
            live_pre._forecast = lam * v + (1 - lam) * live_pre._forecast
        assert cont["retailer"]._forecast == pytest.approx(live_pre._forecast)
        live_post = OrderUpToAgent(theta, lam)
        for v in il["retailer"]:             # weeks 0..2 -> post update of week 2
            live_post._forecast = lam * v + (1 - lam) * live_post._forecast
        assert warm_forecast(theta, lam, il["retailer"]) == pytest.approx(live_post._forecast)

    def test_branch_equals_run_game_on_unbranched_path(self):
        """A branch that orders exactly what the floor would order must cost the
        same as the run_game tail over the same weeks (AUDIT-1 checklist item)."""
        theta, lam = 3.0, 0.5
        demand = [4, 4, 8, 8, 8, 8, 8, 8]
        agents = {r: OrderUpToAgent(theta, lam) for r in ROLES}
        log = run_game(agents, demand, SimConfig(horizon=len(demand)))
        total_tail = sum(w.cost for w in log.weeks[2:])
        # decision week 2: branch from a stepper at week 2 with warm floors
        walk_agents = {r: OrderUpToAgent(theta, lam) for r in ROLES}
        step = GameStepper(demand, SimConfig(horizon=len(demand)))
        il_hist: dict[str, list[int]] = {r: [] for r in ROLES}
        while step.t < 2:
            obs = step.observations()
            for r in ROLES:
                il_hist[r].append(obs[r]["incoming_last"])
            step.advance({r: walk_agents[r](obs[r]) for r in ROLES})
        obs = step.observations()
        for r in ROLES:
            il_hist[r].append(obs[r]["incoming_last"])   # decision week 2
        cont = make_continuation(theta, lam, il_hist, week=2)
        role = "retailer"
        cand = cont[role](obs[role])       # the floor's own order -> unbranched
        j = branch_cost(step, 2, role, cand, cont, demand[2:],
                        candidate_warm=warm_forecast(theta, lam, il_hist[role][:3]))
        assert j == pytest.approx(total_tail, abs=1e-9)


# --------------------------------------------------------------------------- #
# Hand-computed J (integer-exact continuation theta=1, lam=0 -> q = max(0, -IP))
# --------------------------------------------------------------------------- #
class TestHandComputedJ:
    """Demand [4,4,4], H=3, retailer decides at t=0, continuation theta=1 lam=0
    (q = max(0, -IP); nothing ever ships: avail=0 and receipts=0 every week).

    Engine trace (verified against the code, integer-exact):
      candidate 2 — orders(ret,wh,di,fa) [[2,0,0,0],[2,2,0,0],[4,2,2,0]],
                    backlogs        [[4,2,0,0],[8,4,2,0],[12,8,4,2]]
        w0: 2*4+2*2 = 12 ; w1: 2*8+2*4+2*2 = 28 ; w2: 2*12+2*8+2*4+2*2 = 52
        J(a=2) = 92
      candidate 0 — orders [[0,0,0,0],[4,0,0,0],[4,4,0,0]],
                    backlogs [[4,0,0,0],[8,4,0,0],[12,8,4,0]]
        w0: 8 ; w1: 16+8 = 24 ; w2: 24+16+8 = 48 ; J(a=0) = 80
    """

    @pytest.fixture()
    def setup(self):
        demand = [4, 4, 4]
        step = GameStepper(demand, SimConfig(horizon=3))
        il = {r: [0] for r in ROLES}     # week-0 incoming_last = engine init 0
        cont = make_continuation(1.0, 0.0, il, week=0)
        return step, cont, demand, il

    def test_candidate_two_costs_92(self, setup):
        step, cont, demand, il = setup
        j = branch_cost(step, 0, "retailer", 2, cont, demand,
                        candidate_warm=warm_forecast(1.0, 0.0, il["retailer"][:1]))
        assert j == pytest.approx(92.0, abs=1e-9)

    def test_candidate_zero_costs_80(self, setup):
        step, cont, demand, il = setup
        j = branch_cost(step, 0, "retailer", 0, cont, demand,
                        candidate_warm=warm_forecast(1.0, 0.0, il["retailer"][:1]))
        assert j == pytest.approx(80.0, abs=1e-9)

    def test_crn_same_futures_for_all_candidates(self, setup, monkeypatch):
        """Every candidate must be scored on the SAME future set (CRN)."""
        step, cont, demand, _ = setup
        futures = [demand] * 4            # M=4, all the same vector
        seen: list[list] = []
        import agent_bullwhip.decision_labeler as dl
        real = dl.branch_cost

        def spy(*args, **kw):
            seen.append(args[5])          # branch_cost(... future, ...)
            return real(*args, **kw)

        monkeypatch.setattr(dl, "branch_cost", spy)
        score_candidates(step, 0, "retailer", [0, 2, 40], cont, futures,
                         1.0, 0.0, {r: [0] for r in ROLES})
        assert len(seen) == 3 * 4
        assert all(f is futures[i % 4] for i, f in enumerate(seen))  # CRN


# --------------------------------------------------------------------------- #
# Targets, tau, and the run pipeline
# --------------------------------------------------------------------------- #
class TestTargets:
    def test_softmax_minus_j_shape_and_flat(self):
        import math
        p, tau, flat = softmax_minus_j([10.0, 20.0, 30.0], tau_scale=0.5)
        assert not flat
        assert sum(p) == pytest.approx(1.0)
        assert p[0] > p[1] > p[2]          # lower cost -> higher probability
        assert tau == pytest.approx(0.5 * math.sqrt(200 / 3))  # std of [10,20,30]
        p2, tau2, flat2 = softmax_minus_j([5.0, 5.0, 5.0], tau_scale=0.5)
        assert flat2 and p2 == [1 / 3] * 3
        assert tau2 == pytest.approx(0.5 * 1e-3)

    def test_flat_and_edge_reporting(self, tmp_path):
        out = tmp_path / "labels"
        m = run_labeler(40, 2, horizon=8, m_futures=4, tau_scale=0.5,
                        out_prefix=str(out))
        assert "flat_rate" in m and "edge_argmin_rate" in m
        assert 0 <= m["flat_rate"] <= 1 and 0 <= m["edge_argmin_rate"] <= 1
        assert m["continuation_theta"] == 3.5
        assert "tau_rule" in m
        assert (out.with_suffix(".jsonl").exists()
                and out.with_suffix(".npz").exists()
                and out.with_suffix(".manifest.json").exists())


class TestPipeline:
    def test_late_week_exclusion(self):
        tasks = enumerate_states(1, 2, 4, ["poisson"])
        assert len(tasks) == 2 * 4 * (4 - 2)
        assert all(t["week"] < 4 - 2 for t in tasks)

    @pytest.mark.parametrize("bad_seed", [1000, 1015, 1029])
    def test_eval_seed_guard_rejects_reserved_band(self, bad_seed):
        with pytest.raises(ValueError):
            run_labeler(bad_seed, 1, horizon=4, m_futures=2)

    @pytest.mark.parametrize("ok_seed", [999, 1030, 6000])
    def test_eval_seed_guard_accepts_outside_band(self, ok_seed):
        m = run_labeler(ok_seed, 1, horizon=4, m_futures=2)
        assert m["n_states"] == 4 * (4 - 2)

    def test_determinism_same_seed_identical_rows(self, tmp_path):
        o1 = tmp_path / "a"; o2 = tmp_path / "b"
        run_labeler(7, 2, horizon=8, m_futures=4, out_prefix=str(o1))
        run_labeler(7, 2, horizon=8, m_futures=4, out_prefix=str(o2))
        assert o1.with_suffix(".jsonl").read_bytes() == \
            o2.with_suffix(".jsonl").read_bytes()
        m1 = json.loads(o1.with_suffix(".manifest.json").read_text())
        m2 = json.loads(o2.with_suffix(".manifest.json").read_text())
        assert m1["hash"] == m2["hash"]

    def test_determinism_across_process_counts(self, tmp_path):
        """Worker parallelism must not change the output (rngs derive from
        stable strings, never iteration order)."""
        o1 = tmp_path / "a"; o2 = tmp_path / "b"
        run_labeler(9, 2, horizon=8, m_futures=4, processes=1,
                    out_prefix=str(o1))
        run_labeler(9, 2, horizon=8, m_futures=4, processes=2,
                    out_prefix=str(o2))
        assert o1.with_suffix(".jsonl").read_bytes() == \
            o2.with_suffix(".jsonl").read_bytes()

    def test_npz_mirror_matches_jsonl(self, tmp_path):
        import numpy as np
        o = tmp_path / "lbl"
        run_labeler(13, 1, horizon=6, m_futures=3, out_prefix=str(o))
        rows = [json.loads(l) for l in o.with_suffix(".jsonl").read_text().splitlines()]
        z = np.load(o.with_suffix(".npz"))
        assert z["J"].shape == (len(rows), len(ORDER_GRID))
        assert z["p"].shape == (len(rows), len(ORDER_GRID))
        for i, r in enumerate(rows):
            # npz stores J/p as float32 by design (training-side artifact); upcast
            # before comparing and use a float32-safe tolerance. The old abs=1e-5
            # was env-fragile: the hub's pytest passed the same bytes while the
            # cluster's older pytest/numpy scalar-promotion rules failed them.
            # (C4 lesson: never let test tolerances depend on numpy/pytest version.)
            assert list(z["J"][i].astype("float64")) == pytest.approx(r["J"], rel=1e-6, abs=1e-6)
            assert list(z["p"][i].astype("float64")) == pytest.approx(r["p"], rel=1e-6, abs=1e-6)
            assert int(z["week"][i]) == r["week"]
            assert ROLES[int(z["role_idx"][i])] == r["role"]
            assert int(z["argmin"][i]) == r["argmin_idx"]

    def test_state_pool_is_curricula_only_and_engine_ctxs(self, tmp_path):
        """Rows carry real engine observation ctxs (renderer-safe), and only
        the two wave-6 curricula appear."""
        o = tmp_path / "lbl"
        run_labeler(17, 3, horizon=6, m_futures=2, out_prefix=str(o))
        rows = [json.loads(l) for l in o.with_suffix(".jsonl").read_text().splitlines()]
        assert {r["curriculum"] for r in rows} <= {"poisson", "truncnorm"}
        for r in rows:
            assert set(r["ctx"]) == {"role", "t", "on_hand", "backlog",
                                     "outstanding", "incoming_last",
                                     "incoming_now", "last_orders"}
            assert r["ctx"]["role"] == r["role"]
            if r["role"] != "retailer":
                assert r["ctx"]["incoming_now"] is None

# --------------------------------------------------------------------------- #
# Round-2 relabeling: --states-jsonl (DAgger close-the-loop, audit fix 8)
# --------------------------------------------------------------------------- #
class TestStatesRelabel:
    def _collect_dump(self, seed: int = 31, n_eps: int = 1, horizon: int = 8):
        """Fabricate a decision_eval --collect-states dump from a replay."""
        from agent_bullwhip.decision_labeler import (
            _replay_to_week, BehaviorConfig, sample_curriculum_demand_ex)
        from agent_bullwhip.engine import ROLES as R
        import random as _r
        rows = []
        for ep in range(n_eps):
            cur = "poisson" if ep % 2 == 0 else "truncnorm"
            rng = _seeded_rng(f"{seed}:ep:{ep}")
            params, demand = sample_curriculum_demand_ex(rng, cur, horizon)
            from agent_bullwhip.engine import GameStepper
            step = GameStepper(demand, SimConfig(horizon=horizon))
            pol = BehaviorConfig(1.0, 0.0)
            il_hist = {r: [] for r in R}
            w = 0
            while not step.done() and w < horizon - 2:
                obs = step.observations()
                for r in R:
                    il_hist[r].append(obs[r]["incoming_last"])
                for r in R:
                    rows.append({
                        "episode": ep, "week": w, "role": r,
                        "curriculum": cur, "params": params, "ctx": obs[r],
                        "il_hist": {r2: list(il_hist[r2]) for r2 in R},
                        "state": {
                            "t": w,
                            "tiers": {r2: [step.state[r2].oh, step.state[r2].b,
                                           step.state[r2].o] for r2 in R},
                            "inbound": {r2: list(step.inbound[r2]) for r2 in R},
                            "prev_incoming": dict(step.prev_incoming),
                            "order_history": {r2: list(step.order_history[r2])
                                              for r2 in R},
                        },
                        "order": 8,
                    })
                step.advance({r: 8 for r in R})
                w += 1
        return rows

    def test_stepper_reconstruction_matches_live_branch(self):
        """A stepper rebuilt from a collected state must branch identically to
        the live engine at that week (same candidate, same future -> same J)."""
        from agent_bullwhip.decision_labeler import (
            _replay_to_week, BehaviorConfig, branch_cost, make_continuation,
            stepper_from_state, warm_forecast)
        from agent_bullwhip.decision_readout import ORDER_GRID as G
        from agent_bullwhip.engine import ROLES as R

        behavior = BehaviorConfig(0.75, 0.25)
        step, il_hist, params = _replay_to_week(7, 0, 3, "poisson", 8, behavior)
        st = {
            "t": 3,
            "tiers": {r: [step.state[r].oh, step.state[r].b, step.state[r].o]
                      for r in R},
            "inbound": {r: list(step.inbound[r]) for r in R},
            "prev_incoming": dict(step.prev_incoming),
            "order_history": {r: list(step.order_history[r]) for r in R},
        }
        rebuilt = stepper_from_state(st, horizon=8)
        cont = make_continuation(3.5, 0.2, il_hist, 3)
        cand_warm = warm_forecast(3.5, 0.2, il_hist["retailer"][:4])
        future = [8, 8, 8, 8, 8]
        for cand in (0, 20, 40):
            j_live = branch_cost(step, 3, "retailer", cand, cont, future,
                                 candidate_warm=cand_warm)
            j_rebuilt = branch_cost(rebuilt, 3, "retailer", cand, cont, future,
                                    candidate_warm=cand_warm)
            assert j_live == pytest.approx(j_rebuilt, abs=1e-9)

    def test_run_labeler_from_states_deterministic(self, tmp_path):
        rows = self._collect_dump()
        src = tmp_path / "states.jsonl"
        with open(src, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        m1 = run_labeler_from_states(src, seed=6001, horizon=8, m_futures=4,
                                     out_prefix=str(tmp_path / "a"))
        m2 = run_labeler_from_states(src, seed=6001, horizon=8, m_futures=4,
                                     out_prefix=str(tmp_path / "b"))
        assert m1["hash"] == m2["hash"]
        assert m1["n_collected"] == len(rows)
        # late weeks (t >= H-2 = 6,7) excluded by construction of _collect_dump
        assert m1["n_states"] == len(rows)
        assert (tmp_path / "a.jsonl").read_bytes() == \
            (tmp_path / "b.jsonl").read_bytes()
        for r in m1["rows"]:
            assert set(r) >= {"episode", "week", "role", "J", "p", "flat",
                              "argmin_order"}

    def test_states_relabel_eval_seed_guard(self, tmp_path):
        rows = self._collect_dump()
        src = tmp_path / "s.jsonl"
        with open(src, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with pytest.raises(ValueError):
            run_labeler_from_states(src, seed=1005, horizon=8, m_futures=2)
