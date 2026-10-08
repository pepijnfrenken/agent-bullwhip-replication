"""WAVE8 scripts: S_nom construction, oracle replay agreement, F0/F1 smoke, analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agent_bullwhip.decision_readout import get_grid, render_state

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gates_mod():
    return _load("wave8_gates")


@pytest.fixture(scope="module")
def s_nom(gates_mod):
    return gates_mod.build_s_nom()


def test_s_nom_construction(s_nom):
    assert len(s_nom) == 1024
    eps = sorted({r["episode"] for r in s_nom})
    assert eps[0] == 0 and eps[-1] <= 15
    assert {r["seed"] for r in s_nom} <= set(range(9100, 9116))
    assert max(r["week"] for r in s_nom) <= 33
    # stride 2 over (episode, week, role) order -> consecutive kept rows alternate roles pairwise
    assert [r["role"] for r in s_nom[:4]] == ["retailer", "distributor", "retailer", "distributor"]
    for r in s_nom[:200]:
        render_state(r["ctx"])                                  # leak-free engine ctx (raises on a leak)
    assert any(r["ctx"]["incoming_now"] is not None for r in s_nom) and all(
        r["ctx"]["incoming_now"] is None for r in s_nom if r["role"] != "retailer")


def test_s_nom_is_deterministic_and_the_floor_policy_visits_it(gates_mod, s_nom):
    again = gates_mod.build_s_nom()
    assert json.dumps(again[:50], sort_keys=True) == json.dumps(s_nom[:50], sort_keys=True)
    # nominal floor (theta 3.5, lam 0.2): week-0 states are the all-zero starts
    w0 = [r for r in s_nom if r["week"] == 0]
    assert w0 and all(r["ctx"]["outstanding"] == r["ctx"]["outstanding"] for r in w0)


def test_oracle_task_replays_the_same_states(gates_mod, s_nom):
    from agent_bullwhip import decision_labeler as L
    pick = [s_nom[0], s_nom[301], s_nom[900]]
    for r in pick:
        t = {"seed": r["seed"], "ep": 0, "week": r["week"], "role": r["role"],
             "curriculum": r["curriculum"], "m_futures": 2, "theta": L.CONTINUATION_THETA,
             "lam": L.CONTINUATION_LAM, "tau_scale": 0.5, "horizon": 36, "p_floor": 1.0,
             "p_random": 0.0, "theta_range": (3.5, 3.5), "lam_range": (0.2, 0.2), "grid_id": "w8_g26"}
        lab = L._label_state_task(t)
        assert lab["ctx"] == r["ctx"], "oracle replay must see the identical S_nom state"
        assert len(lab["J"]) == 26 and lab["argmin_order"] == get_grid("w8_g26").values[lab["argmin_idx"]]


def test_probe_and_teacher_smoke(tmp_path):
    probe, teacher = _load("wave8_grid_probe"), _load("wave8_teacher_check")
    out = tmp_path / "f0.json"
    probe.main(["--states", "4", "--m-futures", "2", "--out", str(out), "--gates", str(tmp_path / "g.json")])
    res = json.loads(out.read_text())
    assert set(res["stats"]) == {"mixed", "nominal"} and res["verdict"]["pass"] in (True, False)
    assert json.loads((tmp_path / "g.json").read_text())["F0"]["verdict"] == res["verdict"]
    assert res["stats"]["mixed"]["n_states"] == 4
    # a probe seed in the eval band is refused
    with pytest.raises(SystemExit):
        probe.main(["--states", "1", "--label-seed", "1005", "--out", str(tmp_path / "x.json")])
    out1 = tmp_path / "f1.json"
    teacher.main(["--games", "1", "--m-futures", "1", "--out", str(out1)])
    r1 = json.loads(out1.read_text())
    assert r1["seeds"] == [9000] and r1["pi1"][0] > 0 and r1["floor"][0] > 0
    with pytest.raises(SystemExit):
        teacher.main(["--games", "2", "--seed-base", "1000", "--out", str(tmp_path / "y.json")])


def test_permutation_probe_builds_grids_for_the_active_grid():
    pp = _load("probe_permutation_invariance")
    g26 = pp.build_grids(3, 20261006, "w8_g26")
    assert len(g26) == 4 and g26[0] == list(get_grid("w8_g26").values)
    assert all(sorted(g) == g26[0] for g in g26)
    g21 = pp.build_grids(3, 20261006)                           # default = wave-7 grid, unchanged
    assert g21[0] == list(range(0, 41, 2))


# -------------------------------------------------------------- analysis ------
def _arm(path: Path, name: str, costs: list[float], mass=0.95):
    with (path / f"{name}.jsonl").open("w") as f:
        for i, c in enumerate(costs):
            f.write(json.dumps({"run": i, "total_cost": c, "orders": {"retailer": [4, 5]},
                                "failures": {"retailer": 0}}) + "\n")
    (path / f"{name}.summary.json").write_text(json.dumps({"parse_fail": 0, "min_mass": mass}))


def _eval_dir(tmp_path, family="r1", trained_factor=0.7, n_seeds=3):
    e = tmp_path / "eval"
    e.mkdir(parents=True)
    base = [1000.0 + 37 * (i % 11) for i in range(30)]
    for pat, mode in (("noisy", "argmax"), ("step", "sampled")):
        _arm(e, f"w8-untrained-{mode}-{pat}", base)
        for s in range(n_seeds):
            f_ = trained_factor if s < 2 else 1.1
            _arm(e, f"w8-{family}-seed{s}-{mode}-{pat}",
                 [b * f_ - (b - 1000) * 0.4 * (1 if f_ < 1 else 0) for b in base])
    return e


def test_analysis_g1_g4_and_decision_inconclusive_without_preflight(tmp_path):
    an = _load("wave8_analysis")
    e = _eval_dir(tmp_path)
    (tmp_path / "w8").mkdir()
    res = an.analyse(e, tmp_path / "w8", tmp_path / "nofloors.json", tmp_path / "n1", tmp_path / "n2")
    fam = res["families"]["r1"]
    assert fam["g1_arm"] == {"seeds": 3, "passes": 2, "needed": 2, "pass": True}
    assert fam["seeds"][2]["g1"]["pass"] is False and fam["g4_pass_seeds"] == 2
    assert res["decision"]["verdict"] == "INCONCLUSIVE"      # pre-flight F0-F3 not recorded
    assert any("pre-flight" in r for r in res["decision"]["reasons"])
    assert "G1" in an.render(res)


def test_analysis_supporting_and_collapsed_arm_cannot_support(tmp_path):
    an = _load("wave8_analysis")
    e = _eval_dir(tmp_path)
    w8 = tmp_path / "w8"
    w8.mkdir()
    (w8 / "gates.json").write_text(json.dumps({
        "F0": {"verdict": {"pass": True}}, "F1": {"verdict": {"pass": True}},
        "F2": {"contiguous": True, "prompt_sha_equal": True}, "F3": {"H_lo": 1}}))
    res = an.analyse(e, w8, tmp_path / "f.json", tmp_path / "n1", tmp_path / "n2")
    assert res["preflight"]["clean"] and res["decision"]["verdict"].startswith("SUPPORTING")
    for s in range(3):
        (w8 / f"r1-seed{s}-S_nom.json").write_text(json.dumps({"collapse": {"collapsed": True}}))
    res2 = an.analyse(e, w8, tmp_path / "f.json", tmp_path / "n1", tmp_path / "n2")
    assert res2["decision"]["verdict"] == "INCONCLUSIVE"
    assert any("collapsed" in r for r in res2["decision"]["reasons"])


def test_analysis_contradicting_needs_every_rung_and_r2(tmp_path):
    an = _load("wave8_analysis")
    e = tmp_path / "eval"
    e.mkdir()
    base = [1000.0 + 37 * (i % 11) for i in range(30)]
    for ctrl, fams in (("untrained", ("r1", "r2sft")), ("untrained-1.7b", ("r1-1.7b",)),
                       ("untrained-4b", ("r1-4b",))):
        _arm(e, f"w8-{ctrl}-argmax-noisy", base)
        for fam in fams:
            for s in range(2):
                _arm(e, f"w8-{fam}-seed{s}-argmax-noisy", [b * 1.05 for b in base])
    w8 = tmp_path / "w8"
    w8.mkdir()
    (w8 / "gates.json").write_text(json.dumps({
        "F0": {"verdict": {"pass": True}}, "F1": {"verdict": {"pass": True}},
        "F2": {"contiguous": True, "prompt_sha_equal": True}, "F3": {"H_lo": 1}}))
    res = an.analyse(e, w8, tmp_path / "f.json", tmp_path / "n1", tmp_path / "n2")
    assert res["decision"]["verdict"].startswith("CONTRADICTING")
    (e / "w8-r1-4b-seed0-argmax-noisy.jsonl").unlink()
    (e / "w8-r1-4b-seed1-argmax-noisy.jsonl").unlink()
    assert an.analyse(e, w8, tmp_path / "f.json", tmp_path / "n1", tmp_path / "n2")[
        "decision"]["verdict"] == "INCONCLUSIVE"


def test_select_lr_and_sft_lr(tmp_path):
    an = _load("wave8_analysis")
    v = tmp_path / "val"
    v.mkdir()
    for lr, f in (("lr1e-5", 1.0), ("lr5e-5", 0.9)):
        for s in range(3):
            _arm(v, f"w8-val-r1-{lr}-seed{s}", [1000.0 * f + i for i in range(30)])
    r = an.select_lr_from_val(v)
    assert r["winner"] == "lr5e-5" and r["seeds_per_lr"]["lr1e-5"] == [0, 1, 2]
    res = tmp_path / "res"
    for tag, nll in (("lr1e-5", 2.2), ("lr1e-4", 1.9)):
        d = res / f"wave8_r2sft_lrsweep_{tag}"
        d.mkdir(parents=True)
        (d / "train_log.jsonl").write_text(
            json.dumps({"event": "val", "step": 99, "nll": 9.0}) + "\n"
            + json.dumps({"event": "val", "step": 499, "nll": nll}) + "\n")
    assert an.select_sft_lr(res)["winner"] == "lr1e-4"
    # a third sweep lr must be seen, not silently dropped (the arm swept {1e-5,1e-4,2e-4})
    d = res / "wave8_r2sft_lrsweep_lr2e-4"
    d.mkdir(parents=True)
    (d / "train_log.jsonl").write_text(json.dumps({"event": "val", "step": 499, "nll": 0.5}) + "\n")
    r = an.select_sft_lr(res)
    assert r["winner"] == "lr2e-4" and r["n_candidates"] == 3


def test_analysis_runs_on_the_real_checkout_without_w8_files(capsys):
    an = _load("wave8_analysis")
    assert an.main([]) == 0
    assert "Decision" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["sample", "oracle", "control", "collapse-check", "collapse-final",
                                 "perm-check", "report"])
def test_every_gates_subcommand_builds_its_parser(gates_mod, cmd):
    with pytest.raises(SystemExit) as e:
        gates_mod.main([cmd, "--help"])
    assert e.value.code == 0


def test_committed_preflight_artifacts_are_self_consistent(gates_mod, s_nom):
    import hashlib
    from agent_bullwhip import decision_gates as G
    w8 = ROOT / "results" / "wave8"
    if not (w8 / "gates.json").exists():
        pytest.skip("pre-flight artifacts not in this checkout")
    g = json.loads((w8 / "gates.json").read_text())
    # S_nom regenerates byte-identically from code and matches the recorded sha
    raw = "".join(json.dumps(r, sort_keys=True) + "\n" for r in s_nom)
    assert hashlib.sha256(raw.encode()).hexdigest() == g["S_nom"]["sha256"]
    assert (w8 / "S_nom.jsonl").read_text() == raw
    # F0 verdict recomputes from the recorded stats; F1 from the recorded paired costs
    f0 = json.loads((w8 / "f0.json").read_text())
    assert G.f0_gate(f0["stats"]) == f0["verdict"] == g["F0"]["verdict"]
    assert f0["states_per_regime"] >= 1000 and f0["grid"] == "w8_g26b"
    f1 = json.loads((w8 / "f1.json").read_text())
    assert G.f1_gate(f1["pi1"], f1["floor"]) == f1["verdict"] == g["F1"]["verdict"]
    assert f1["seeds"] == list(range(9000, 9030))
    # the pre-registered grid's failing read is kept next to it
    pre = json.loads((w8 / "f0_w8_g26_prereg.json").read_text())
    assert pre["verdict"]["pass"] is False and g["F0_prereg_w8_g26"]["verdict"]["pass"] is False
    # oracle: modal share and its Wilson bound recompute from the argmin letters
    orc = json.loads((w8 / "S_nom_oracle.json").read_text())
    top = max(set(orc["argmin"]), key=orc["argmin"].count)
    assert orc["modal_letter"] == top and orc["m_star"] == orc["argmin"].count(top) / 1024
    assert orc["m_hi"] == pytest.approx(G.wilson_upper(orc["argmin"].count(top), 1024))


def test_probe_refuses_a_grid_above_the_reference_max(tmp_path):
    probe = _load("wave8_grid_probe")
    with pytest.raises(SystemExit, match="exceeds the reference max"):
        probe.main(["--states", "1", "--grid", "w8_r3c", "--out", str(tmp_path / "x.json")])
