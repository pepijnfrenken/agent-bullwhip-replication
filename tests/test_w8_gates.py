"""WAVE8 gate math (agent_bullwhip/decision_gates.py) on synthetic AND real wave-7 shapes."""
from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

from agent_bullwhip import decision_gates as G

ROOT = Path(__file__).resolve().parents[1]
K = 26


def _softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = sum(e)
    return [x / z for x in e]


def _untrained(n=400, seed=0):
    """Untrained-readout shape: a near-constant letter prior (peak at letter 4) + noise."""
    rng = random.Random(seed)
    return [_softmax([(2.5 if a == 4 else 0.0) + rng.gauss(0, 0.35) for a in range(K)]) for _ in range(n)]


def _oracle(n=400, seed=1):
    rng = random.Random(seed)
    return [min(int(abs(rng.gauss(7, 5))), K - 1) for _ in range(n)]       # modal share ~ 0.1


@pytest.fixture(scope="module")
def frozen():
    return G.freeze_control_stats(_untrained(), _oracle())


def _peaked(letter, mass):
    rest = (1 - mass) / (K - 1)
    return [mass if a == letter else rest for a in range(K)]


# -------------------------------------------------------------- basics ------
def test_percentile_nearest_rank():
    v = [5, 1, 4, 2, 3]
    assert G.percentile(v, 0.05) == 1 and G.percentile(v, 0.95) == 5 and G.percentile(v, 0.5) == 3
    assert G.percentile(list(range(1, 101)), 0.95) == 95


def test_wilson_upper_closed_form_and_symmetry():
    n, z = 1024, G.Z95
    assert G.wilson_upper(0, n) == pytest.approx(z * z / (n + z * z))
    assert G.wilson_upper(n, n) == 1.0
    assert G.wilson_upper(512, 1024) > 0.5
    assert G.wilson_upper(100, 1024) > 100 / 1024 and G.wilson_upper(100, 1024) < 0.13


def test_mutual_information_prior_only_vs_state_dependent():
    prior = [_peaked(3, 0.7)] * 50
    assert G.mutual_information(prior) == pytest.approx(0.0, abs=1e-12)
    onehot = [[1.0 if a == s % 5 else 0.0 for a in range(K)] for s in range(50)]
    assert G.mutual_information(onehot) == pytest.approx(math.log(5))


def test_bootstrap_is_deterministic_and_small_for_a_prior(frozen):
    P = _untrained(100, seed=3)
    a = G.bootstrap_mi_percentile(P, B=50)
    assert a == G.bootstrap_mi_percentile(P, B=50)
    assert 0 <= a < 0.5


def test_freeze_control_stats_shape(frozen):
    assert frozen["n_labels"] == K and frozen["H_lo"] < frozen["H_hi"]
    assert frozen["m_star"] <= frozen["m_hi"] <= 1
    assert frozen["I0"] <= frozen["I_hi0"] + 1e-9 or frozen["I_hi0"] > 0
    assert "collapsed iff" in frozen["rule"]


# ------------------------------------------------------ collapse gate --------
def test_letter_lock_is_collapsed(frozen):
    res = G.collapse_gate([_peaked(20, 0.97)] * 200, frozen)
    assert res["collapsed"] and res["branch"] == "low" and res["modal_argmax_share"] == 1.0


def test_diffuse_state_blind_is_collapsed(frozen):
    res = G.collapse_gate([[1 / K] * K] * 200, frozen)
    assert res["collapsed"] and res["branch"] == "high"


def test_confident_state_dependent_policy_is_not_collapsed(frozen):
    # low entropy (below H_lo) but 12 different argmax letters and high I: the goal, not collapse
    P = [_peaked(2 + s % 12, 0.9) for s in range(240)]
    res = G.collapse_gate(P, frozen)
    assert res["median_entropy"] < frozen["H_lo"]
    assert not res["collapsed"] and res["I"] > frozen["I_hi0"]


def test_untrained_control_itself_reads_as_not_diffuse(frozen):
    res = G.collapse_gate(_untrained(300, seed=9), frozen)
    assert res["branch"] != "high"


def test_collapse_gate_threshold_edges():
    fz = {"H_lo": 1.0, "H_hi": 3.0, "m_hi": 0.30, "I_hi0": 0.05}
    def rows_with_entropy(h_target, letters):
        # two-level distribution with entropy ~h_target via bisection on the peak mass
        lo, hi = 1 / K, 1 - 1e-9
        for _ in range(60):
            mid = (lo + hi) / 2
            if G.entropy_nats(_peaked(0, mid)) > h_target:
                lo = mid
            else:
                hi = mid
        m = (lo + hi) / 2
        return [_peaked(letters[s % len(letters)], m) for s in range(100)]
    # entropy just below H_lo, one letter (share 1 > m_hi) -> collapsed
    assert G.collapse_gate(rows_with_entropy(0.99, [5]), fz)["collapsed"]
    # entropy just ABOVE H_lo with a single letter and I = 0 -> neither branch fires
    assert not G.collapse_gate(rows_with_entropy(1.01, [5]), fz)["collapsed"]
    # entropy above H_hi and state-blind -> collapsed; just below H_hi -> not
    assert G.collapse_gate(rows_with_entropy(3.01, [5]), fz)["collapsed"]
    assert not G.collapse_gate(rows_with_entropy(2.99, [5]), fz)["collapsed"]
    # low entropy, many letters (share 0.1 <= m_hi) and I > I_hi0 -> not collapsed
    assert not G.collapse_gate(rows_with_entropy(0.5, list(range(10))), fz)["collapsed"]


def test_consecutive_trips():
    f = lambda *x: [{"collapsed": v} for v in x]
    assert G.consecutive_trips(f(False, True, True)) and not G.consecutive_trips(f(True, False))


# --------------------------------------------------- permutation gate --------
def test_permutation_gate_synthetic():
    vals = [[0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 8, 6]]
    r = G.permutation_gate(vals, 0.9)
    assert r["invariant"] == 3 and r["invariance_rate"] == 0.75 and not r["pass"]
    assert G.permutation_gate(vals, 0.0)["pass"]


@pytest.mark.parametrize("arm", ["rlcd-seed0", "sft-seed1", "untrained"])
def test_permutation_gate_reproduces_wave7_zero_reads(arm):
    p = ROOT / "results" / "eval" / f"w7-{arm}-perm.json"
    if not p.exists():
        pytest.skip("wave-7 perm json not in checkout")
    d = json.loads(p.read_text())
    r = G.permutation_gate(d["values"], d["gate"] or 0.9)
    assert r["invariance_rate"] == pytest.approx(d["invariance_rate"])
    assert r["invariant"] == d["invariant"]


# ------------------------------------------------------- F0 / F1 -------------
def _J_rows(n, opt_values, ref):
    """Convex J around each state's optimum (cost 100 + |v-opt|)."""
    return [[100.0 + abs(v - o) for v in ref] for o in opt_values]


def test_f0_stats_regret_and_top_edge():
    ref = list(range(0, 201, 2))
    grid = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 38, 44, 52, 62, 76, 94,
            116, 140, 160]
    inside = G.f0_stats(_J_rows(4, [8, 20, 30, 94], ref), grid, ref)
    assert inside["regret_mean_rel"] == 0.0 and inside["top_argmax_share"] == 0.0
    outside = G.f0_stats(_J_rows(2, [200, 200], ref), grid, ref)       # optimum beyond the cap
    assert outside["regret_mean_rel"] == pytest.approx(40 / 100) and outside["top_argmax_share"] == 1.0
    assert outside["top_mass"] > 0.5
    odd = G.f0_stats(_J_rows(2, [36, 36], ref), grid, ref)             # 36 is in the reference, not the grid
    assert odd["regret_mean_rel"] == pytest.approx(2 / 100)   # best grid points 34/38 cost 102 vs 100


def test_f0_gate_thresholds_edges():
    good = {"regret_mean_rel": 0.005, "regret_p90_rel": 0.02, "top_mass": 0.06, "top_argmax_share": 0.08}
    nominal = {"regret_mean_rel": 0.005, "regret_p90_rel": 0.02, "top_mass": 0.03, "top_argmax_share": 0.03}
    assert G.f0_gate({"mixed": good, "nominal": nominal})["pass"]
    for k, bump in (("regret_mean_rel", 0.0051), ("regret_p90_rel", 0.0201), ("top_mass", 0.0601),
                    ("top_argmax_share", 0.0801)):
        assert not G.f0_gate({"mixed": dict(good, **{k: bump}), "nominal": nominal})["pass"]
    assert not G.f0_gate({"mixed": good, "nominal": dict(nominal, top_mass=0.0301)})["pass"]
    assert G.f0_gate({"mixed": good})["pass"] is None


def test_f1_gate():
    assert G.f1_gate([10, 20], [11, 25])["pass"] and G.f1_gate([10, 20], [11, 25])["wins"] == 2
    assert not G.f1_gate([30, 20], [10, 25])["pass"]


# -------------------------------------------------- performance gates --------
def _w7(path):
    spec = importlib.util.spec_from_file_location("w7a", ROOT / "scripts" / "wave7_analysis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_g1_matches_wave7_c1_on_real_wave7_files():
    w7 = _w7(None)
    e = ROOT / "results" / "eval"
    tr, ct = e / "w7-rlcd-seed0-argmax-noisy.jsonl", e / "w7-untrained-argmax-noisy.jsonl"
    if not (tr.exists() and ct.exists()):
        pytest.skip("wave-7 eval JSONL not in checkout")
    t, u = w7.run_costs(w7.load_arm(tr)), w7.run_costs(w7.load_arm(ct))
    pa = w7.paired(t, u)
    pa["median_trained"] = w7.median(list(t.values()))
    pa["median_untrained"] = w7.median(list(u.values()))
    ref = w7.gate_c1(pa)
    mine = G.g1_seed(t, u)
    assert mine["reduction_of_medians"] == pytest.approx(ref["reduction_of_medians"])
    assert mine["median_of_reductions"] == pytest.approx(ref["median_of_reductions"])
    assert mine["wins"] == pa["wins"] and mine["sign_p"] == pytest.approx(pa["sign_p"])
    assert mine["pass"] == ref["pass"] is False                       # wave 7: C1 failed on every trained arm


def test_g1_pass_and_arm_rule():
    u = {i: 1000.0 + i for i in range(30)}
    good = {i: 800.0 + i for i in range(30)}
    bad = {i: 1100.0 + i for i in range(30)}
    g_ok, g_bad = G.g1_seed(good, u), G.g1_seed(bad, u)
    assert g_ok["pass"] and g_ok["wins"] == 30 and not g_bad["pass"]
    assert G.g1_arm([g_ok, g_ok, g_bad])["pass"] and not G.g1_arm([g_ok, g_bad, g_bad])["pass"]
    assert G.g1_arm([g_ok, g_bad])["needed"] == 2


def test_g4_triplet_and_g5():
    u = [1000.0 + 300 * (i % 7) for i in range(30)]
    t = [x * 0.7 - (x - 1000) * 0.4 for x in u]            # lower mean, lower CV, lower max
    g4 = G.g4_triplet(t, u)
    assert g4["pass"] and g4["mean_reduction"] >= 0.2 and g4["cv_reduction"] >= 0.25
    assert not G.g4_triplet(u, u)["pass"]
    rng = random.Random(0)
    a = [rng.gauss(1000, 100) for _ in range(30)]
    assert G.g5_agreement([a, a, a])["agree"]
    const = G.g5_agreement([[100.0] * 30, [200.0] * 30, [100.0] * 30])
    assert not const["agree"] and const["mechanically_disagreeing"] and const["zero_variance_seeds"] == [0, 1, 2]


def test_g7():
    assert G.g7_parse(0, 0.95, True)["pass"] and not G.g7_parse(1, 0.99, False)["pass"]
    assert not G.g7_parse(0, 0.85, True)["pass"] and G.g7_parse(0, 0.85, False)["pass"]


def test_calibration_hand_cases():
    conf, ok = [0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]
    assert G.ece(conf, ok) == pytest.approx(0.1) and G.brier(conf, ok) == pytest.approx(0.01)
    assert G.auroc(conf, ok) == 1.0 and G.auroc([0.5] * 4, ok) == 0.5
    assert math.isnan(G.auroc([0.1, 0.2], [1, 1]))
    P = [_peaked(3, 0.8), _peaked(5, 0.6)]
    J = [[100 + abs(a - 3) * 50 for a in range(K)], [100 + abs(a - 9) * 50 for a in range(K)]]
    ct = G.calibration_table(P, J)
    assert ct["accuracy"] == 0.5 and ct["n"] == 2


def test_select_lr_pooled_mean_and_tie_default():
    r = G.select_lr({"lr1e-5": {0: [100, 120], 1: [110]}, "lr5e-5": {0: [90, 95], 1: [92]}}, "lr1e-5")
    assert r["winner"] == "lr5e-5"
    assert G.select_lr({"lr1e-5": {0: [100]}, "lr5e-5": {0: [100]}}, "lr1e-5")["winner"] == "lr1e-5"


def test_update_json_key_freezes_unless_forced(tmp_path):
    p = tmp_path / "g.json"
    G.update_json_key(p, "F3", {"a": 1}, force=False)
    with pytest.raises(SystemExit):
        G.update_json_key(p, "F3", {"a": 2}, force=False)
    G.update_json_key(p, "F3", {"a": 2}, force=True)
    G.update_json_key(p, "F0", {"b": 1})
    assert json.loads(p.read_text()) == {"F0": {"b": 1}, "F3": {"a": 2}}
