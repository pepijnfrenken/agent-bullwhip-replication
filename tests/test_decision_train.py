"""Wave 7 P3: decision_train — data cursor + checkpoint (hub, stub policy) and
the torch learning/RLCD math (jev venv, importorskip).

The hub half follows the grpo_train pattern: the loop is torch-free and any
DecisionPolicy stub drives it. The torch half runs only where torch + peft +
transformers exist (class-scoped _Policy, small steps to keep the host memory
tight).
"""
from __future__ import annotations

import json
import math
import random

import pytest

from agent_bullwhip.decision_labeler import softmax_minus_j
from agent_bullwhip.decision_readout import ORDER_GRID, render_prompt
from agent_bullwhip.decision_train import (
    StepResult, load_labels, parse_args, permute_grid_and_target,
    run_loop, save_checkpoint, load_resume_state, train_val_split,
)
from agent_bullwhip.engine import ROLES


def synth_rows(n: int = 24) -> list[dict]:
    """Renderer-valid labeler rows (engine ctx shapes, real curricula-ish J)."""
    rows = []
    for i in range(n):
        role = ROLES[i % 4]
        ctx = {"role": role, "t": i % 24,
               "on_hand": (i * 7) % 40, "backlog": (i * 3) % 25,
               "outstanding": (i * 5) % 30, "incoming_last": (i * 2) % 20,
               "incoming_now": (i * 3) % 20 if role == "retailer" else None,
               "last_orders": [(i + k) % 30 for k in range(6)]}
        # unique (episode, week) per row so batch identity is injective
        i_ep, i_wk = i // 24, i % 24
        j = [float(abs(k - (i % 21)) * 5 + (i % 7)) for k in range(21)]
        p, tau, _ = softmax_minus_j(j, 0.5)
        argmin = min(range(21), key=j.__getitem__)
        rows.append({"episode": i_ep, "week": i_wk, "role": role,
                     "curriculum": "poisson", "ctx": ctx, "J": j, "p": p,
                     "tau": tau, "argmin_idx": argmin,
                     "argmin_order": ORDER_GRID[argmin]})
    return rows


def write_labels(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return str(path)


def stub_args(tmp_path, **over):
    argv = {"labels": "x", "out": str(tmp_path), "phase": "sft",
            "steps": 12, "batch": 8, "lr": 1e-4, "val_frac": 0.1,
            "save_every": 5, "seed": 7, "beta_kl": 0.02, "rlcd_sample_k": 0,
            "permute_options": 1, "model": "stub", "epochs": None,
            "base_adapter": None, "resume": False, "device": None,
            "dtype": None, "max_length": 1024, "revision": None}
    argv.update(over)
    return parse_args(["--labels", argv["labels"], "--phase", argv["phase"],
                       "--steps", str(argv["steps"]), "--batch", str(argv["batch"]),
                       "--lr", str(argv["lr"]), "--val-frac", str(argv["val_frac"]),
                       "--save-every", str(argv["save_every"]),
                       "--seed", str(argv["seed"]), "--out", argv["out"]])


class RecorderStub:
    """DecisionPolicy stub: records batch identities, no torch."""
    model = None
    tok = None

    def __init__(self):
        self.batches: list[list[tuple]] = []
        self.calls = 0
        self.saved_optimizer = False

    def _ids(self, rows):
        return [(r["episode"], r["week"], r["role"]) for r in rows]

    def sft_step(self, rows, targets, lr, seed, epoch):
        self.batches.append(self._ids(rows))
        self.calls += 1
        ent = -sum(x * math.log(x) for p in targets for x in p if x > 0)
        return StepResult(loss=1.0 - self.calls * 1e-4,
                          mean_target_entropy=ent / len(targets),
                          train_top1=0.5)

    def rlcd_step(self, rows, j_vals, lr, beta_kl, sample_k, seed, step):
        self.batches.append(self._ids(rows))
        self.calls += 1
        return StepResult(loss=2.0, loss_kl=0.01)

    def evaluate(self, rows):
        return {"nll": 2.5, "top1": 0.4, "mean_regret": 12.0}

    def check_finite(self):
        return True

    def save_optimizer(self, path):
        self.saved_optimizer = True
        return True

    def load_optimizer(self, path, lr):
        return True


# --------------------------------------------------------------------------- #
# Hub: cursor determinism, resume continuity, checkpoint metadata
# --------------------------------------------------------------------------- #
class TestCursorAndCheckpoint:
    def test_train_val_split_deterministic(self):
        rows = synth_rows(100)
        v1 = train_val_split(rows, 0.05)
        v2 = train_val_split(rows, 0.05)
        assert v1 == v2 and len(v1) == 5 and v1 == [0, 20, 40, 60, 80]

    def test_permute_grid_and_target_consistency(self):
        rng = random.Random(3)
        p = [0.0] * 21
        p[9] = 0.8
        p[3] = 0.2
        perm, tgt = permute_grid_and_target(p, rng)
        assert sorted(perm) == list(range(21))
        # target label i must equal the source prob of the value shown at i
        assert tgt == [p[i] for i in perm]
        assert sum(tgt) == pytest.approx(1.0)
        assert 0.8 in tgt

    def test_load_labels_and_sha(self, tmp_path):
        rows = synth_rows(10)
        path = write_labels(tmp_path / "l.jsonl", rows)
        loaded, sha = load_labels(path)
        assert len(loaded) == 10 and len(sha) == 64
        (tmp_path / "bad.jsonl").write_text("not json\n")
        with pytest.raises(ValueError):
            load_labels(tmp_path / "bad.jsonl")

    def test_run_loop_resume_continuity_exact(self, tmp_path):
        """The audit §4 P2 fixed-data-cursor property: a run resumed at step k
        sees EXACTLY the batches the uninterrupted run sees from step k on."""
        rows = synth_rows(30)
        labels = write_labels(tmp_path / "l.jsonl", rows)
        full = RecorderStub()
        args = stub_args(tmp_path / "full", steps=12, batch=8, val_frac=0.05)
        args.labels = labels
        run_loop(full, args, tmp_path / "full" / "train_log.jsonl",
                 tmp_path / "full" / "checkpoint")

        resumed = RecorderStub()
        args2 = stub_args(tmp_path / "res", steps=12, batch=8, val_frac=0.05)
        args2.labels = labels
        run_loop(resumed, args2, tmp_path / "res" / "train_log.jsonl",
                 tmp_path / "res" / "checkpoint", start_step=5)
        # full run steps 5..11 == resumed steps 0..6 (same epoch/perm math)
        assert resumed.batches == full.batches[5:]
        # batch-level determinism across full process runs too
        full2 = RecorderStub()
        args3 = stub_args(tmp_path / "full2", steps=12, batch=8, val_frac=0.05)
        args3.labels = labels
        run_loop(full2, args3, tmp_path / "full2" / "train_log.jsonl",
                 tmp_path / "full2" / "checkpoint")
        assert full2.batches == full.batches
        # every train row appears exactly once per epoch (no dupes, no gaps)
        flat = [x for b in full.batches[:math.ceil(28 / 8)] for x in b]
        assert len(flat) == len(set(flat)) == 28

    def test_checkpoint_roundtrip_metadata(self, tmp_path):
        rows = synth_rows(8)
        labels = write_labels(tmp_path / "l.jsonl", rows)
        _, sha = load_labels(labels)
        pol = RecorderStub()
        args = stub_args(tmp_path / "out", steps=8)
        ck = tmp_path / "out" / "checkpoint"
        payload = save_checkpoint(pol, ck, ck.parent / "state.json", step=3,
                                  seed=11, args=args, labels_sha=sha,
                                  phase="sft", has_torch=lambda: False)
        assert payload["step"] == 3 and payload["seed"] == 11
        assert "adapter_saved" in payload and payload["adapter_saved"] is False
        step, loaded = load_resume_state(tmp_path / "out")
        assert step == 4 and loaded["labels_sha"] == sha

    def test_run_loop_sigterm_checkpoint_cadence(self, tmp_path):
        """SIGTERM (stop event) checkpoints at the next save boundary and exits
        before consuming the rest of the run."""
        import threading
        rows = synth_rows(24)
        labels = write_labels(tmp_path / "l.jsonl", rows)
        pol = RecorderStub()
        args = stub_args(tmp_path / "out", steps=9, save_every=3)
        args.labels = labels
        stop = threading.Event()
        stop.set()                              # kill signal already pending
        run_loop(pol, args, tmp_path / "out" / "train_log.jsonl",
                 tmp_path / "out" / "checkpoint", start_step=0, stop=stop)
        log = (tmp_path / "out" / "train_log.jsonl").read_text()
        assert '"event": "sigterm"' in log
        assert pol.calls == 1                   # stopped at the step-0 boundary
        assert (tmp_path / "out" / "state.json").exists()

    def test_phase_rlcd_requires_base_adapter(self):
        import argparse
        with pytest.raises(SystemExit):
            parse_args(["--labels", "x", "--phase", "rlcd"])

# --------------------------------------------------------------------------- #
# Torch-backed (jev venv): real _Policy learning, resume roundtrip, RLCD math
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def torch_ok():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return True


@pytest.fixture(scope="module")
def sft_policy(torch_ok):
    from agent_bullwhip.decision_train import _Policy
    pol = _Policy("Qwen/Qwen3-0.6B", device="cpu", phase="sft",
                  max_length=1024, seed=5, permute_options=True)
    return pol


class TestSftTorch:
    def test_sft_loss_decreases_on_toy_targets(self, sft_policy, tmp_path):
        rows = synth_rows(24)
        for r in rows:
            k = r["argmin_idx"]
            r["p"] = [0.6 / 20 if i != k else 0.6 for i in range(21)]
        losses = []
        for step in range(20):
            batch = rows[(step * 4) % 24:(step * 4) % 24 + 4]
            if len(batch) < 4:
                batch = rows[:4]
            out = sft_policy.sft_step(batch, [r["p"] for r in batch],
                                      lr=5e-4, seed=5, epoch=step)
            losses.append(out.loss)
        assert losses[-1] < 0.70 * losses[0], losses[:3] + losses[-3:]
        assert losses[0] > math.log(21) - 1.0          # near-uniform start

    def test_checkpoint_resume_roundtrip_torch(self, torch_ok, tmp_path):
        """Resume loads adapter + optimizer and continues the exact loss trace
        of an uninterrupted run (audit checklist: hub CPU resume drill)."""
        from agent_bullwhip.decision_train import _Policy, save_checkpoint
        import argparse
        rows = synth_rows(24)
        rng = random.Random(2)
        for r in rows:
            p = [0.0] * 21
            p[r["argmin_idx"]] = 0.7
            rest = 0.3 / 20
            r["p"] = [rest if i != r["argmin_idx"] else 0.7 for i in range(21)]
        def trace(pol, start, steps):
            out = []
            for step in range(start, start + steps):
                batch = rows[(step * 8) % 24:(step * 8) % 24 + 8]
                if len(batch) < 8:
                    batch = rows[:8]
                out.append(pol.sft_step(batch, [r["p"] for r in batch],
                                        lr=2e-4, seed=5, epoch=step).loss)
            return out
        pol = _Policy("Qwen/Qwen3-0.6B", device="cpu", phase="sft",
                      max_length=1024, seed=5, permute_options=False)
        t_full = trace(pol, 0, 6)
        t_cont = trace(pol, 6, 3)
        import gc
        del pol
        gc.collect()
        args = argparse.Namespace(model="Qwen/Qwen3-0.6B", lr=2e-4, batch=8,
                                  epochs=None, permute_options=0, beta_kl=None,
                                  seed=5, val_frac=0.05)
        ck = tmp_path / "ck"
        save_checkpoint(pol, ck, tmp_path / "state.json", step=5, seed=5,
                        args=args, labels_sha="x" * 64, phase="sft")
        # resume: fresh policy on the saved adapter + optimizer
        pol2 = _Policy("Qwen/Qwen3-0.6B", device="cpu", phase="sft",
                       adapter_dir=str(tmp_path / "ck-5"), max_length=1024,
                       seed=5, permute_options=False)
        pol2.load_optimizer(tmp_path / "ck-5" / "optimizer.pt", lr=2e-4)
        t_res = trace(pol2, 6, 3)
        assert t_res == pytest.approx(t_cont, abs=1e-5)
        assert len(t_full) == 6 and len(t_res) == 3

    def test_rlcd_exact_pg_moves_mass_to_low_cost(self, torch_ok, tmp_path):
        """Exact expected-cost PG must RAISE pi on the low-J action (sign fix:
        J is a cost; the naive J - mean ascends it)."""
        from agent_bullwhip.decision_train import _Policy
        rows = synth_rows(2)
        for i, r in enumerate(rows):
            j = [100.0 + (20 - i) * 5] * 21
            j[0] = 30.0 + i * 10          # action A (order 0) is cheapest
            r["J"] = j
        # run rlcd on a bare SFT-initialized policy (pi_sft warm from itself).
        pol = _Policy("Qwen/Qwen3-0.6B", device="cpu", phase="sft",
                      max_length=1024, seed=5, permute_options=False)
        pol.phase = "rlcd"
        pol._pi_sft_cache = {}
        import torch
        with torch.no_grad():
            prompts = [render_prompt(r["ctx"], pol.tok) for r in rows]
            logits = pol._label_logits(pol._tokenize(prompts))
            before = logits.softmax(-1)[:, 0].tolist()
        out = pol.rlcd_step(rows, [r["J"] for r in rows], lr=3e-3,
                            beta_kl=0.0, sample_k=0, seed=5, step=0)
        assert math.isfinite(out.loss)
        logits = pol._label_logits(pol._tokenize([render_prompt(r["ctx"], pol.tok)
                                                  for r in rows]))
        after = logits.softmax(-1)[:, 0].tolist()
        assert all(a > b for a, b in zip(after, before)), (before, after)

    def test_rlcd_sampled_arm_loo_runs_and_is_finite(self, torch_ok, tmp_path):
        """The K-sampling LOO arm (public recipe) must run and stay finite on
        the same data. Its v1 (self-included baseline) would shrink the
        gradient — the test pins LOO-excluding-self by construction."""
        from agent_bullwhip.decision_train import _Policy
        rows = synth_rows(2)
        for r in rows:
            j = [float(50 + (10 * i)) for i in range(21)]
            j[r["argmin_idx"]] = 10.0
            r["J"] = j
        pol = _Policy("Qwen/Qwen3-0.6B", device="cpu", phase="sft",
                      max_length=1024, seed=5, permute_options=False)
        pol.phase = "rlcd"
        pol._pi_sft_cache = {}
        out = pol.rlcd_step(rows, [r["J"] for r in rows], lr=1e-3,
                            beta_kl=0.01, sample_k=8, seed=5, step=0)
        assert math.isfinite(out.loss) and math.isfinite(out.loss_kl)
        assert out.grad_norm >= 0


# --------------------------------------------------------------------------- #
# decision_eval: protocol arms + state dump (hub-safe pieces)
# --------------------------------------------------------------------------- #
class TestEvalArmsHub:
    def test_make_demands_step_argmax_is_single_game(self):
        from agent_bullwhip.decision_eval import make_demands
        d = make_demands("step", "argmax", runs=30, horizon=8, seed_base=1000)
        assert len(d) == 1            # audit P0: step-argmax is n=1
        assert d[0] == [4, 4, 4, 4, 8, 8, 8, 8]

    def test_make_demands_step_sampled_is_n_runs(self):
        from agent_bullwhip.decision_eval import make_demands
        d = make_demands("step", "sampled", runs=30, horizon=8, seed_base=1000)
        assert len(d) == 30 and all(x == d[0] for x in d)

    def test_make_demands_noisy_seeded_distinct(self):
        from agent_bullwhip.decision_eval import make_demands
        d = make_demands("noisy", "argmax", runs=6, horizon=8, seed_base=1000)
        assert len(d) == 6
        assert len({tuple(x) for x in d}) == 6   # genuinely different paths
        assert all(len(x) == 8 for x in d)

    def test_eval_record_schema_matches_wave6_consumer(self, tmp_path):
        """eval-analyze.load_arm reads run/config/model/tag/total_cost/orders/
        failures; a wave-7-style record must load and yield complete recs."""
        import importlib.util as _ilu
        from agent_bullwhip.engine import ROLES as R
        rec = {"run": 0, "config": "baseline", "model": "qwen3-0.6b-untrained",
               "tag": "qwen3-0.6b-readout-untrained-argmax-step",
               "total_cost": 1234.0,
               "orders": {r: [0, 2] for r in R},
               "backlogs": {r: [0, 0] for r in R},
               "failures": {r: 0 for r in R},
               "calls": {r: 2 for r in R},
               "tokens": {"prompt": 900, "completion": 0},
               "prompt_hash_dups": 0,
               "traces": {r: [] for r in R},
               "tool_traces": {r: [] for r in R}}
        out = tmp_path / "arm.jsonl"
        out.write_text(json.dumps(rec) + "\n")
        from pathlib import Path as _P
        spec = _ilu.spec_from_file_location(
            "eval_analyze", _P(__file__).resolve().parents[1] /
            "hpc-umbrella" / "eval-analyze.py")
        ea = _ilu.module_from_spec(spec)
        spec.loader.exec_module(ea)
        complete, failed, dpr = ea.load_arm(out, n_weeks=2, n_roles=4)
        assert len(complete) == 1 and len(failed) == 0 and dpr == 8
        cs = ea.cost_stats(complete)
        assert cs["mean"] == 1234.0
        assert ea.parse_failure_rate(complete, dpr)[0] == 0.0

    def test_sample_orders_deterministic_and_seed_dependent(self):
        pytest.importorskip("torch")
        from agent_bullwhip.decision_eval import sample_orders
        probs = [[1.0, 0.0] + [0.0] * 19, [0.0, 1.0] + [0.0] * 19,
                 [0.5, 0.5] + [0.0] * 19]
        a = sample_orders(probs, [0, 1, 2], seed_base=1000)
        b = sample_orders(probs, [0, 1, 2], seed_base=1000)
        assert a == b
        assert a[(0, 0)] == 0 and a[(1, 0)] == 1       # deterministic peeks
        c = sample_orders(probs, [0, 1, 2], seed_base=1001)
        assert c != b or c[(2, 0)] != b[(2, 0)]        # different seeds -> diff

    def test_collect_state_dump_hub_shape(self, tmp_path, monkeypatch):
        """Collection rows must carry the full engine snapshot + params the
        labeler --states-jsonl needs (validated against engine fields)."""
        from agent_bullwhip.decision_eval import collect_policy_states
        from agent_bullwhip.engine import ROLES as R

        class FakeModel:
            class _Tok:
                @staticmethod
                def apply_chat_template(messages, **kw):
                    return "prompt"
            tokenizer = _Tok()
            def readout_batch(self, prompts):
                return [{"argmax_order": 8, "tokens": 10} for _ in prompts]

        fake = FakeModel()
        out = tmp_path / "states.jsonl"
        n = collect_policy_states(fake, n_episodes=2, seed=7000,
                                  curricula=["poisson"], horizon=6,
                                  out_path=out)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert n == len(rows) > 0
        for row in rows:
            assert set(row) >= {"episode", "week", "role", "curriculum",
                                "params", "ctx", "il_hist", "state", "order"}
            st = row["state"]
            assert set(st) >= {"t", "tiers", "inbound", "prev_incoming",
                               "order_history"}
            assert {r2 for r2 in st["tiers"]} == set(R)
            assert st["t"] == row["week"]
            assert len(st["inbound"]["retailer"]) == 2   # shipment_delay pipe
            assert all(len(v) == 6 for v in st["order_history"].values()) is None \
                or all(len(v) == row["week"] for v in st["order_history"].values())
