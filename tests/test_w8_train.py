"""WAVE8 training machinery: flags, permutation/J/anchor alignment, micro-batching,
seed bands, merge, and the round orchestrator (recording runner; no torch)."""
from __future__ import annotations

import json
import math
import random
import threading
from pathlib import Path

import pytest

from agent_bullwhip import decision_rounds as dr
from agent_bullwhip import decision_train as dt
from agent_bullwhip.decision_labeler import _seeded_rng

from test_decision_train import RecorderStub, stub_args, synth_rows, write_labels


# ------------------------------ flags -------------------------------------- #
def _args(*extra):
    return dt.parse_args(["--labels", "x", *extra])


def test_legacy_defaults_are_wave7():
    a = _args("--phase", "sft")
    assert (a.grid, a.init, a.anchor, a.permute_p, a.brier_weight, a.micro_batch) == \
        ("w7_u21", "adapter", "adapter", 1.0, 0.0, 0)
    assert a.permute_p_explicit is False
    assert dt._w8_policy_kwargs(a) == {}                      # stubs see the wave-7 call
    assert _args("--phase", "sft", "--permute-options", "0").permute_p == 0.0


def test_rlcd_requires_base_adapter_unless_fresh():
    with pytest.raises(SystemExit):
        _args("--phase", "rlcd")
    a = _args("--phase", "rlcd", "--init", "fresh", "--anchor", "base", "--grid", "w8_g26",
              "--permute-p", "0.5")
    assert a.base_adapter is None and a.permute_p_explicit and a.permute_p == 0.5
    assert dt._w8_policy_kwargs(a) == {"grid_id": "w8_g26", "init": "fresh", "anchor": "base",
                                       "permute_p": 0.5}
    with pytest.raises(SystemExit):
        _args("--phase", "rlcd", "--init", "fresh", "--base-adapter", "x")
    with pytest.raises(SystemExit):
        _args("--phase", "sft", "--permute-p", "1.5")
    with pytest.raises(SystemExit):
        _args("--phase", "rlcd", "--base-adapter", "x", "--anchor", "external")
    ok = _args("--phase", "rlcd", "--base-adapter", "x", "--anchor", "external",
               "--anchor-table", "t.json")
    assert ok.anchor == "external"


# --------------------- permutation / J / anchor alignment ------------------ #
def test_row_permutation_p1_reproduces_wave7_derivation():
    for ep, wk, role in [(0, 3, "retailer"), (7, 20, "factory")]:
        n = 21
        old = list(range(n))
        _seeded_rng(f"5:perm:2:{ep}:{wk}:{role}").shuffle(old)
        assert dt.row_permutation(1.0, 5, f"2:{ep}:{wk}:{role}", n) == old
    assert dt.row_permutation(0.0, 5, "k", 26) is None


def test_row_permutation_bernoulli_rate_and_validity():
    hits = [dt.row_permutation(0.5, 1, f"rlcd:3:{i}:5:retailer", 26) for i in range(2000)]
    frac = sum(h is not None for h in hits) / 2000
    assert 0.45 < frac < 0.55
    assert all(sorted(h) == list(range(26)) for h in hits if h is not None)
    assert dt.row_permutation(0.5, 1, "rlcd:3:1:5:retailer", 26) == \
        dt.row_permutation(0.5, 1, "rlcd:3:1:5:retailer", 26)       # deterministic


def _rlcd_exact_loss(probs, j):
    """pure-python mirror of rlcd_step's exact-gradient term: sum_a pi(a) (J-mean)/max(std,1e-3)."""
    n = len(j)
    m = sum(j) / n
    std = max((sum((x - m) ** 2 for x in j) / (n - 1)) ** 0.5, 1e-3)
    return sum(p * (x - m) / std for p, x in zip(probs, j))


def test_permuting_letters_and_J_together_keeps_the_rlcd_objective():
    rng = random.Random(0)
    j = [rng.uniform(100, 900) for _ in range(26)]
    w = [rng.random() for _ in range(26)]
    probs = [x / sum(w) for x in w]
    perm = list(range(26))
    rng.shuffle(perm)
    # label i of the permuted rendering shows value perm[i]; the model's prob for that label is
    # the canonical prob of value perm[i] when it reads values (an invariant policy)
    assert _rlcd_exact_loss(dt.permute_row(probs, perm), dt.permute_row(j, perm)) == \
        pytest.approx(_rlcd_exact_loss(probs, j))
    # permuting only J is NOT the same objective (the bug the spec's unit test guards)
    assert _rlcd_exact_loss(probs, dt.permute_row(j, perm)) != pytest.approx(
        _rlcd_exact_loss(probs, j))


def test_permute_grid_and_target_26_and_legacy_signature():
    p = [0.0] * 26
    p[25] = 1.0
    perm, tgt = dt.permute_grid_and_target(p, random.Random(1), 26)
    assert sorted(perm) == list(range(26)) and tgt[perm.index(25)] == 1.0
    perm21, _ = dt.permute_grid_and_target([0.0] * 21, random.Random(1))
    assert len(perm21) == 21


def test_j_std_clamped_and_microbatches():
    assert dt.j_std_clamped([5.0] * 26) and not dt.j_std_clamped([1.0, 2.0, 9.0])
    assert dt.split_microbatches(10, 4) == [range(0, 4), range(4, 8), range(8, 10)]
    assert dt.split_microbatches(10, 0) == [range(10)] == dt.split_microbatches(10, 10)
    assert dt.split_microbatches(0, 4) == []


def test_rlcd_log_has_std_j_clamped_frac(tmp_path):
    rows = synth_rows(16)
    for r in rows[:4]:
        r["J"] = [7.0] * 21                    # flat states: clamped
    labels = write_labels(tmp_path / "l.jsonl", rows)
    args = stub_args(tmp_path / "o", steps=2, batch=16, val_frac=0.0001)
    args.labels = labels
    args.phase = "rlcd"
    dt.run_loop(RecorderStub(), args, tmp_path / "o" / "train_log.jsonl",
                tmp_path / "o" / "checkpoint", has_torch=lambda: False)
    recs = [json.loads(l) for l in (tmp_path / "o" / "train_log.jsonl").read_text().splitlines()]
    steps = [r for r in recs if "step" in r and "event" not in r]
    assert steps and all("std_j_clamped_frac" in r for r in steps)
    assert any(r["std_j_clamped_frac"] > 0 for r in steps)


def test_external_anchor_table(tmp_path):
    rows = synth_rows(6)
    labels = write_labels(tmp_path / "l.jsonl", rows)
    _, sha = dt.load_labels(labels)
    ext = tmp_path / "ext.json"
    ext.write_text(json.dumps({"labels_sha": sha, "pi": {dt.pi_sft_key(r): [1 / 21] * 21 for r in rows}}))

    class NoCompute:
        def compute_pi_sft(self, *a):
            raise AssertionError("external anchor must not compute")
    out = tmp_path / "o"
    out.mkdir()
    tab = dt.prepare_pi_sft(NoCompute(), rows, out, labels_sha=sha, batch=4, resuming=False,
                            log_path=out / "log.jsonl", anchor="external", external_table=ext)
    assert len(tab) == 6 and json.loads((out / "pi_sft.json").read_text())["anchor"] == "external"
    with pytest.raises(SystemExit):       # resume under a different anchor kind
        dt.prepare_pi_sft(NoCompute(), rows, out, labels_sha=sha, batch=4, resuming=True,
                          log_path=out / "log.jsonl", anchor="base")
    ext.write_text(json.dumps({"labels_sha": "bad", "pi": {}}))
    out2 = tmp_path / "o2"
    out2.mkdir()
    with pytest.raises(SystemExit):
        dt.prepare_pi_sft(NoCompute(), rows, out2, labels_sha=sha, batch=4, resuming=False,
                          log_path=out2 / "log.jsonl", anchor="external", external_table=ext)


# ------------------------------ seeds / merge ------------------------------- #
def test_round_seed_bands_are_disjoint_from_every_other_band():
    other = (set(range(1000, 1030)) | set(range(4000, 4120)) | set(range(6000, 6003))
             | set(range(7000, 7003)) | set(range(9000, 9030)) | set(range(9100, 9116))
             | set(range(9500, 9530)))
    seen = []
    for s in range(3):
        for r in range(1, 4):
            seen += list(dr.round_seeds(s, r).values())
    assert len(seen) == len(set(seen)) == 36
    assert not (set(seen) & other)
    assert dr.VAL_SEED_BASE == 9500


def test_merge_label_files_unique_keys_and_stable_sha(tmp_path):
    a, b = synth_rows(8), synth_rows(8)            # identical keys: would collide unmerged
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for p, rows in ((pa, a), (pb, b)):
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "m.jsonl"
    m1 = dr.merge_label_files([pa, pb], out)
    rows, sha = dt.load_labels(out)
    keys = [dt.pi_sft_key(r) for r in rows]
    assert len(set(keys)) == 16 and m1["sha256"] == sha and m1["n"] == 16
    assert dr.merge_label_files([pa, pb], tmp_path / "m2.jsonl")["sha256"] == sha
    with pytest.raises(ValueError):                # same source twice in one namespace
        pc = tmp_path / "c.jsonl"
        pc.write_text(pa.read_text() + pa.read_text())
        dr.merge_label_files([pc], tmp_path / "m3.jsonl")


def test_consecutive_trips():
    f = lambda *xs: [{"collapsed": x} for x in xs]
    assert dr.consecutive_trips(f(False, True, True))
    assert not dr.consecutive_trips(f(True, False, True)) and not dr.consecutive_trips(f(True))


def test_lr_tag():
    assert dr.lr_tag(1e-5) == "lr1e-5" and dr.lr_tag(5e-5) == "lr5e-5"


# ------------------------------ orchestrator ------------------------------- #
class FakeRunner:
    """Executes nothing; materialises the files each stage would leave behind."""

    def __init__(self, collapse=(False, False, False), fail_at=None, partial_at=None,
                 stop_event=None, stop_at=None):
        self.calls, self.collapse, self.fail_at, self.partial_at = [], list(collapse), fail_at, partial_at
        self.stop, self.stop_at = stop_event or threading.Event(), stop_at
        self.partial_done = False

    @staticmethod
    def _opt(argv, flag):
        return argv[argv.index(flag) + 1] if flag in argv else None

    def __call__(self, argv):
        mod = next((a for a in argv if a.startswith("agent_bullwhip.")), argv[2] if len(argv) > 2 else "")
        key = f"{mod.split('.')[-1]}"
        self.calls.append((key, argv))
        n = len(self.calls)
        if self.stop_at == n:
            self.stop.set()
            return 143
        if key == "decision_eval" and "--collect-states" in argv:
            out = Path(self._opt(argv, "--collect-states"))
            off = int(self._opt(argv, "--collect-offset"))
            out.write_text("".join(json.dumps({"episode": off + e, "week": w, "role": "retailer"}) + "\n"
                                   for e in range(2) for w in range(2)))
        elif key == "decision_labeler":
            out = Path(self._opt(argv, "--out") + ".jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps({"episode": e, "week": w, "role": "factory"}) + "\n"
                                   for e in range(2) for w in range(2)))
        elif key == "decision_train":
            if self.fail_at == "train":
                return 1
            if self.partial_at and not self.partial_done:
                self.partial_done = True
                return 75
            od = Path(self._opt(argv, "--out"))
            (od / "checkpoint-169").mkdir(parents=True, exist_ok=True)
            (od / "state.json").write_text(json.dumps({"step": 169, "checkpoint_dir": "checkpoint-169"}))
        elif "wave8_gates.py" in " ".join(argv):
            r = int(Path(self._opt(argv, "--out")).stem[-1])
            Path(self._opt(argv, "--out")).write_text(json.dumps({"collapsed": self.collapse[r - 1]}))
        return 0


def _plan(tmp_path, **kw):
    return dr.RoundPlan(arm=kw.pop("arm", "r1"), seed=kw.pop("seed", 0), lr=kw.pop("lr", 1e-5),
                        results=str(tmp_path / "results"), **kw)


def test_full_r1_run_stage_order_and_paths(tmp_path):
    plan = _plan(tmp_path)
    runner = FakeRunner()
    assert dr.run_plan(plan, runner) == 0
    keys = [k for k, _ in runner.calls]
    per_round = ["decision_eval", "decision_eval", "decision_labeler", "decision_labeler",
                 "decision_train", "wave8_gates.py"]
    # stage commands: collect x2, label x2, train, gate per round; then validation eval
    assert [k if k != "wave8_gates.py" else k for k in keys[:5]] == per_round[:5]
    assert keys.count("decision_train") == 3 and keys.count("decision_eval") == 3 * 2 + 1
    trains = [a for k, a in runner.calls if k == "decision_train"]
    assert "--init" in trains[0] and trains[0][trains[0].index("--init") + 1] == "fresh"
    assert all(a[a.index("--anchor") + 1] == "base" for a in trains)
    # rounds 2/3 init from the previous round's FINAL checkpoint, resolved at run time
    assert trains[1][trains[1].index("--base-adapter") + 1].endswith("round1/checkpoint-169")
    assert trains[2][trains[2].index("--base-adapter") + 1].endswith("round2/checkpoint-169")
    # the train argv is accepted by decision_train's own parser
    for a in trains:
        ns = dt.parse_args(a[4:])
        assert ns.phase == "rlcd" and ns.steps == 170 and ns.permute_p == 0.5 and ns.grid == "w8_g26b"
    val = runner.calls[-1][1]
    assert val[val.index("--pattern") + 1] == "curriculum" and val[val.index("--seed-base") + 1] == "9500"
    assert val[val.index("--adapter") + 1].endswith("round3/checkpoint-169")
    assert (plan.arm_dir / "DONE").exists()
    merged = (plan.work(1) / "round_labels.jsonl").read_text().splitlines()
    assert len(merged) == 8                       # 4 policy + 4 behaviour rows, namespaced
    assert len(dr.read_jsonl(plan.arm_dir / "collapse_reads.jsonl")) == 3


def test_resume_after_sigterm_partial_train_skips_done_stages(tmp_path):
    plan = _plan(tmp_path)
    r1 = FakeRunner(partial_at=True)
    assert dr.run_plan(plan, r1) == dr.EX_TEMPFAIL
    done = dr.read_state(plan.state_path)["completed"]
    assert done == ["r1:collect_argmax", "r1:collect_sampled", "r1:concat_states",
                    "r1:label_policy", "r1:label_behavior", "r1:merge"]
    r2 = FakeRunner()
    assert dr.run_plan(plan, r2) == 0
    keys = [k for k, _ in r2.calls]
    assert keys[0] == "decision_train"            # nothing before train re-ran
    assert "r1:train" in dr.read_state(plan.state_path)["completed"]


def test_sigterm_flag_stops_with_75_and_keeps_state(tmp_path):
    plan = _plan(tmp_path)
    stop = threading.Event()
    runner = FakeRunner(stop_event=stop, stop_at=3)          # SIGTERM lands in stage 3's child
    rc = dr.run_plan(plan, runner, stop)
    assert rc == dr.EX_TEMPFAIL and len(runner.calls) == 3
    st = dr.read_state(plan.state_path)["completed"]
    # the third runner call (label_policy) was interrupted: earlier stages + the in-process
    # concat are recorded, the interrupted stage is not and re-runs on resume
    assert st == ["r1:collect_argmax", "r1:collect_sampled", "r1:concat_states"]


def test_collapse_twice_stops_the_job_without_failing(tmp_path):
    plan = _plan(tmp_path)
    runner = FakeRunner(collapse=(True, True, False))
    assert dr.run_plan(plan, runner) == 0
    assert (plan.arm_dir / "STOPPED_collapse.json").exists()
    assert not (plan.arm_dir / "DONE").exists()
    assert [k for k, _ in runner.calls].count("decision_train") == 2


def test_child_failure_propagates_and_does_not_advance(tmp_path):
    plan = _plan(tmp_path)
    assert dr.run_plan(plan, FakeRunner(fail_at="train")) == 1
    assert "r1:train" not in dr.read_state(plan.state_path)["completed"]


def test_r2rl_uses_sft_start_and_frozen_external_anchor(tmp_path):
    with pytest.raises(ValueError):
        _plan(tmp_path, arm="r2rl")
    plan = _plan(tmp_path, arm="r2rl", base_adapter="results/wave8_r2sft_seed0/checkpoint-2826",
                 lr=1e-5)
    runner = FakeRunner()
    # anchor-table stage is the module's own CLI; the fake runner materialises its output
    runner_call = runner.__call__

    def with_anchor(argv):
        if "anchor-table" in argv:
            Path(argv[argv.index("--out") + 1]).write_text("{}")
            runner.calls.append(("anchor", argv))
            return 0
        return runner_call(argv)
    assert dr.run_plan(plan, with_anchor) == 0
    trains = [a for k, a in runner.calls if k == "decision_train"]
    assert len(trains) == 3
    for a in trains:
        ns = dt.parse_args(a[4:])
        assert ns.anchor == "external" and ns.anchor_table.endswith("anchor_table.json")
        assert ns.init == "adapter"
    assert trains[0][trains[0].index("--base-adapter") + 1] == "results/wave8_r2sft_seed0/checkpoint-2826"
    anchors = [a for k, a in runner.calls if k == "anchor"]
    assert all(a[a.index("--adapter") + 1] == "results/wave8_r2sft_seed0/checkpoint-2826"
               for a in anchors)                          # anchor = the SFT adapter in EVERY round
    first_collect = next(a for k, a in runner.calls if k == "decision_eval")
    assert first_collect[first_collect.index("--adapter") + 1].endswith("checkpoint-2826")


def test_sft_argv_is_accepted_by_decision_train():
    a = dr.sft_argv("Qwen/Qwen3-0.6B", "c1899de2", 1, 1e-4, labels="l.jsonl")
    ns = dt.parse_args(a[4:])
    assert (ns.phase, ns.steps, ns.permute_p, ns.save_every, ns.grid, ns.seed) == \
        ("sft", 2827, 0.5, 250, "w8_g26b", 1)


def test_cli_flags_exist_in_target_modules():
    src = {m: Path(__import__("agent_bullwhip." + m, fromlist=["x"]).__file__).read_text()
           for m in ("decision_eval", "decision_labeler")}
    plan = _plan(Path("/tmp/x"))
    for st in dr.round_stages(plan, 1):
        if not st.argv:
            continue
        mod = next((a.split(".")[-1] for a in st.argv if a.startswith("agent_bullwhip.decision_")), None)
        if mod in src:
            for a in st.argv:
                if a.startswith("--"):
                    assert f'"{a}"' in src[mod], (mod, a)
