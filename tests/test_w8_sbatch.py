"""hpc-umbrella/{cpu,gpu}-wave8-*.sbatch: the REAL files under bash with the REAL config.sh pins
(only paths overridden) and a recording `python3` shim — array mapping, gating, resume exit
codes, rung selection, eval tag/adapter naming. No Slurm, no torch."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KITSRC = ROOT / "hpc-umbrella"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

PY_STUB = r'''#!/bin/bash
# records every module/script call; materialises the files the real stages would leave
if [ "$1" = "-" ] || [ "$1" = "-c" ]; then exec "$W8_PY" "$@"; fi
case " $* " in
  *" agent_bullwhip.decision_rounds "*|*" scripts/wave8_"*|*" agent_bullwhip.decision_eval "*|*" scripts/wave8_analysis.py "*)
     echo "$*" >> "$W8_CALLS"
     out=""; prev=""
     for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
     case " $* " in
       *"decision_rounds sft "*)
          mkdir -p "$out"; echo '{"checkpoint_dir": "checkpoint-2826", "step": 2826}' > "$out/state.json";;
       *"select-lr "*) echo '{"winner": "lr5e-5"}' > "$out";;
     esac
     exit "${W8_FAIL_RC:-0}";;
  *) exec "$W8_PY" "$@";;
esac
'''


def _setup(tmp_path, *, gates=True, f3=True, env=None, sbatch="gpu-wave8-train.sbatch", task=0):
    kit, repo, bin_ = tmp_path / "kit", tmp_path / "repo", tmp_path / "bin"
    for d in (kit, repo, bin_):
        d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    # a REAL commit: the commit gate compares rev-parse output, so the fixture needs a real HEAD
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)
    repo_head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    shutil.copy(KITSRC / sbatch, kit)
    (kit / "config.sh").write_text(
        f'. "{KITSRC / "config.sh"}"\nREPO_DIR="{repo}"\nRESULTS_DIR="{repo}/results"\n'
        f'LOG_DIR="{tmp_path / "logs"}"\nbw_load_env() {{ :; }}\n')
    py = bin_ / "python3"
    py.write_text(PY_STUB)
    py.chmod(0o755)
    w8 = repo / "results" / "wave8"
    w8.mkdir(parents=True)
    if gates:
        (w8 / "gates.json").write_text(json.dumps({
            "F0": {"verdict": {"pass": True}}, "F1": {"verdict": {"pass": True}}}))
    for rung in ("gates_1.7b.json", "gates_4b.json") if f3 else ():
        (w8 / rung).write_text(json.dumps({"F2": {"prompt_sha_equal": True}, "F3": {"H_lo": 1}}))
    if f3:
        g = json.loads((w8 / "gates.json").read_text()) if gates else {}
        g.update({"F2": {"prompt_sha_equal": True}, "F3": {"H_lo": 1}})
        (w8 / "gates.json").write_text(json.dumps(g))
    calls = tmp_path / "calls.log"
    calls.touch()
    e = {k: v for k, v in os.environ.items() if not k.startswith(("W8_", "SLURM_", "EXPECTED_COMMIT"))}
    e.update({"PATH": f"{bin_}{os.pathsep}{e['PATH']}", "SLURM_SUBMIT_DIR": str(kit),
              "SLURM_ARRAY_TASK_ID": str(task), "SLURM_CPUS_PER_TASK": "3",
              "W8_CALLS": str(calls), "W8_PY": sys.executable,
              # config.sh refuses to run without a cluster user; the tests supply a fake one so
              # no account identifier is needed to run the suite
              "TUE_USER": "cluster-user",
              # the gate is part of the product now: default the pin to the fixture's own HEAD so
              # every non-gate test still exercises it; a test that wants a refusal overrides this
              "EXPECTED_COMMIT": repo_head})
    e.update(env or {})
    return kit, repo, e, calls


def _run(kit, sbatch, e):
    return subprocess.run(["bash", str(kit / sbatch)], env=e, capture_output=True, text=True,
                          timeout=120, cwd=kit)


def _opt(line, flag):
    toks = line.split()
    return toks[toks.index(flag) + 1]


@pytest.mark.parametrize("task,lr,seed", [(0, "1e-5", 0), (2, "1e-5", 2), (3, "5e-5", 0), (5, "5e-5", 2)])
def test_r1_array_index_maps_to_lr_and_seed(tmp_path, task, lr, seed):
    kit, repo, e, calls = _setup(tmp_path, task=task)
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 1
    L = lines[0]
    assert "decision_rounds run --arm r1 " in L
    assert _opt(L, "--seed") == str(seed) and _opt(L, "--lr") == lr
    assert _opt(L, "--model") == "Qwen/Qwen3-0.6B" and _opt(L, "--gates") == "results/wave8/gates.json"
    assert _opt(L, "--rounds") == "3" and _opt(L, "--steps-per-round") == "170"
    assert "--rung-suffix=" in L and _opt(L, "--processes") == "3"


def test_rung_selection_passes_model_revision_suffix_gates(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, task=1, env={"W8_RUNG": "1.7b", "W8_LRS": "5e-5"})
    assert _run(kit, "gpu-wave8-train.sbatch", e).returncode == 0
    L = calls.read_text().splitlines()[0]
    assert _opt(L, "--model") == "Qwen/Qwen3-1.7B"
    assert _opt(L, "--revision") == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert "--rung-suffix=-1.7b" in L and _opt(L, "--gates") == "results/wave8/gates_1.7b.json"
    assert _opt(L, "--lr") == "5e-5" and _opt(L, "--seed") == "1"
    kit2, _, e2, calls2 = _setup(tmp_path / "b", task=1, env={"W8_RUNG": "4b", "W8_LRS": "5e-5",
                                                              "W8_SEEDS": "2", "W8_MICRO_BATCH": "8"})
    assert _run(kit2, "gpu-wave8-train.sbatch", e2).returncode == 0
    L = calls2.read_text().splitlines()[0]
    assert _opt(L, "--model") == "Qwen/Qwen3-4B" and _opt(L, "--micro-batch") == "8"
    assert _opt(L, "--seed") == "1" and _opt(L, "--revision").startswith("1cfa9a72")


def test_refuses_without_preflight_and_never_calls_python_stages(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, gates=False, f3=False)
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode != 0 and "pre-flight" in (p.stdout + p.stderr) and calls.read_text() == ""
    kit, repo, e, calls = _setup(tmp_path / "b", f3=False)
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode != 0 and "F2/F3" in (p.stdout + p.stderr) and calls.read_text() == ""
    kit, repo, e, calls = _setup(tmp_path / "c")
    g = repo / "results/wave8/gates.json"
    d = json.loads(g.read_text())
    d["F1"]["verdict"]["pass"] = False
    g.write_text(json.dumps(d))
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode != 0 and "F1" in (p.stdout + p.stderr) and calls.read_text() == ""


def test_commit_gate_and_exit_codes(tmp_path):
    kit, _, e, calls = _setup(tmp_path, env={"EXPECTED_COMMIT": "deadbeef"})
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 1 and calls.read_text() == "" and "EXPECTED_COMMIT" in p.stderr
    # a SHORT pin of the right commit must be accepted (regression: job 407040 died on
    # 'EXPECTED_COMMIT=af1d6d0 but repo HEAD=af1d6d0d6ec...' — the gate now rev-parses the pin)
    kit, repo, e, calls = _setup(tmp_path / "short", env={"EXPECTED_COMMIT": "SHORT"})
    e["EXPECTED_COMMIT"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True, check=True).stdout.strip()
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    kit, _, e, calls = _setup(tmp_path / "b", env={"W8_FAIL_RC": "75"})
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 75 and "EXIT 75" in p.stdout
    kit, _, e, calls = _setup(tmp_path / "c", env={"W8_FAIL_RC": "3"})
    assert _run(kit, "gpu-wave8-train.sbatch", e).returncode == 3
    kit, _, e, calls = _setup(tmp_path / "d", task=9)           # lr index outside W8_LRS
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 1 and "outside W8_LRS" in p.stderr


def test_r2_runs_sft_then_gate_reads_then_rl_from_the_sft_final_checkpoint(tmp_path):
    # default (no W8_R2_RL): SFT + gate reads only — RL is gated behind W8_R2_RL=1 (0fa8ba9)
    kit, repo, e, calls = _setup(tmp_path, task=1, env={"W8_ARM": "r2", "W8_R2_LR": "1e-4"})
    (repo / "results/wave8_labels").mkdir()
    (repo / "results/wave8_labels/labels.jsonl").write_text("{}\n")
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    sft, gates = calls.read_text().splitlines()
    assert "decision_rounds sft " in sft and _opt(sft, "--seed") == "1" and _opt(sft, "--steps") == "2827"
    assert _opt(sft, "--out") == "results/wave8_r2sft_seed1" and _opt(sft, "--lr") == "1e-4"
    assert "decision_rounds sft-gates " in gates and _opt(gates, "--sft-dir") == "results/wave8_r2sft_seed1"
    assert "r2rl" not in calls.read_text()                       # gated off by default
    # W8_R2_RL=1: the RL rounds anchored to the frozen SFT final checkpoint
    kit, repo, e, calls = _setup(tmp_path / "rl", task=1, env={"W8_ARM": "r2", "W8_R2_LR": "1e-4",
                                                               "W8_R2_RL": "1"})
    (repo / "results/wave8_labels").mkdir()
    (repo / "results/wave8_labels/labels.jsonl").write_text("{}\n")
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    sft, gates, rl = calls.read_text().splitlines()
    assert "decision_rounds run --arm r2rl " in rl
    assert _opt(rl, "--base-adapter") == "results/wave8_r2sft_seed1/checkpoint-2826"
    assert _opt(rl, "--lr") == "1e-4"
    # r2 without its lr is refused before any stage
    kit, _, e, calls = _setup(tmp_path / "b", env={"W8_ARM": "r2"})
    p = _run(kit, "gpu-wave8-train.sbatch", e)
    assert p.returncode != 0 and calls.read_text() == ""
    # the selector returns a TAG ('lr2e-4'); the sbatch must accept it and the plain float (seam
    # regression: W8_R2_LR=lr2e-4 used to hit argparse's float() parsing and die mid-run)
    for tag in ("lr2e-4", "2e-4"):
        kit, repo, e, calls = _setup(tmp_path / f"tag-{tag}", task=0,
                                     env={"W8_ARM": "r2", "W8_R2_LR": tag})
        (repo / "results/wave8_labels").mkdir()
        (repo / "results/wave8_labels/labels.jsonl").write_text("{}\n")
        p = _run(kit, "gpu-wave8-train.sbatch", e)
        assert p.returncode == 0, p.stdout + p.stderr
        assert _opt(calls.read_text().splitlines()[0], "--lr") == "2e-4"


def test_r2_sweep_two_lrs_with_validation_every_100(tmp_path):
    for task, lr in ((0, "1e-5"), (1, "1e-4")):
        kit, repo, e, calls = _setup(tmp_path / str(task), task=task, env={"W8_ARM": "r2sweep"})
        (repo / "results/wave8_labels").mkdir()
        (repo / "results/wave8_labels/labels.jsonl").write_text("{}\n")
        assert _run(kit, "gpu-wave8-train.sbatch", e).returncode == 0
        L = calls.read_text().splitlines()[0]
        assert _opt(L, "--lr") == lr and _opt(L, "--steps") == "500" and _opt(L, "--val-every") == "100"
        assert _opt(L, "--out") == f"results/wave8_r2sft_lrsweep_lr{lr}"


def test_eval_sbatch_tags_adapters_and_lr_selection(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, sbatch="gpu-wave8-eval.sbatch",
                                 env={"W8_EVAL_SEEDS": "0 1"})
    band = repo / "results/wave8_val"
    band.mkdir(parents=True)
    for s in (0, 1):                                    # select-lr reads the band first
        (band / f"w8-val-r1-lr5e-5-seed{s}.jsonl").write_text("{}\n")
    for s in (0, 1):
        d = repo / f"results/wave8_r1_lr5e-5_seed{s}_round3"
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"checkpoint_dir": "checkpoint-169"}))
    p = _run(kit, "gpu-wave8-eval.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    text = calls.read_text()
    for tag in ("w8-untrained-argmax-step", "w8-untrained-sampled-step", "w8-untrained-argmax-noisy",
                "w8-r1-seed0-argmax-noisy", "w8-r1-seed1-sampled-step"):
        assert f"--tag {tag} " in text, tag
    assert "select-lr" in text and "--family r1" in text
    seed0 = [l for l in text.splitlines() if "w8-r1-seed0-argmax-noisy" in l][0]
    assert _opt(seed0, "--adapter") == "results/wave8_r1_lr5e-5_seed0_round3/checkpoint-169"
    assert "--traces 1" in seed0 and "--grid w8_g26" in seed0
    assert f"w8-r1-seed0-perm.json" in text and "collapse-final" in text
    # a missing adapter dir is fatal
    shutil.rmtree(repo / "results/wave8_r1_lr5e-5_seed1_round3")
    kit2, repo2, e2, calls2 = _setup(tmp_path / "b", sbatch="gpu-wave8-eval.sbatch",
                                     env={"W8_EVAL_SEEDS": "0", "W8_LR": "1e-5"})
    p2 = _run(kit2, "gpu-wave8-eval.sbatch", e2)
    assert p2.returncode == 1 and "missing train output" in p2.stderr
    assert "select-lr" not in calls2.read_text()                 # W8_LR override skips selection


def test_eval_refuses_without_a_validation_band_before_any_python(tmp_path):
    """Select-lr SystemExits on an empty band; the sbatch must refuse in seconds instead.

    Regression for the 2026-10-08 job 405888: the control evals ran for 6 minutes, then the job
    died on 'no w8-val-*.jsonl under results/wave8_val' with nothing pointing at the train array.
    """
    kit, repo, e, calls = _setup(tmp_path, sbatch="gpu-wave8-eval.sbatch")
    (repo / "results/wave8_val").mkdir(parents=True)          # exists but empty
    p = _run(kit, "gpu-wave8-eval.sbatch", e)
    assert p.returncode == 1
    assert "no validation band" in p.stderr
    assert "W8_LR=" in p.stderr                               # the workaround is spelled out
    assert calls.read_text() == ""                            # nothing ran: no control, no select-lr


def test_eval_with_an_explicit_lr_needs_no_band(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, sbatch="gpu-wave8-eval.sbatch",
                                 env={"W8_EVAL_SEEDS": "0", "W8_LR": "1e-5"})
    d = repo / "results/wave8_r1_lr1e-5_seed0_round3"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"checkpoint_dir": "checkpoint-169"}))
    p = _run(kit, "gpu-wave8-eval.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "select-lr" not in calls.read_text()               # override skips selection entirely


def test_eval_notes_a_single_lr_band(tmp_path):
    kit, repo, e, _ = _setup(tmp_path, sbatch="gpu-wave8-eval.sbatch",
                             env={"W8_EVAL_SEEDS": "0"})
    band = repo / "results/wave8_val"
    band.mkdir(parents=True)
    (band / "w8-val-r1-lr1e-5-seed0.jsonl").write_text("{}\n")   # one lr only
    d = repo / "results/wave8_r1_lr5e-5_seed0_round3"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"checkpoint_dir": "checkpoint-169"}))
    p = _run(kit, "gpu-wave8-eval.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "covers 1 lr" in p.stderr                          # non-fatal, but stated


def test_eval_r2_only_arms_do_not_require_the_r1_band(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, sbatch="gpu-wave8-eval.sbatch",
                                 env={"W8_EVAL_ARMS": "r2sft", "W8_R2_LR": "1e-4",
                                      "W8_EVAL_SEEDS": "0"})
    d = repo / "results/wave8_r2sft_seed0"
    (d / "checkpoint-2826").mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"checkpoint_dir": "checkpoint-2826"}))
    p = _run(kit, "gpu-wave8-eval.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "w8-r2sft-seed0-argmax-step" in calls.read_text()


def test_eval_rung_suffix_in_control_tags(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, sbatch="gpu-wave8-eval.sbatch",
                                 env={"W8_RUNG": "1.7b", "W8_EVAL_ARMS": "none-x"})
    p = _run(kit, "gpu-wave8-eval.sbatch", e)
    assert p.returncode == 2                                      # unknown arm refused...
    assert "w8-untrained-1.7b-argmax-step" in calls.read_text()   # ...after the rung's control ran


def test_preflight_sbatch_refuses_without_hub_artifacts_and_runs_f2_f3(tmp_path):
    kit, repo, e, calls = _setup(tmp_path, sbatch="gpu-wave8-preflight.sbatch", f3=False)
    p = _run(kit, "gpu-wave8-preflight.sbatch", e)
    assert p.returncode == 1 and "missing results/wave8/S_nom.jsonl" in p.stderr
    w8 = repo / "results/wave8"
    for f in ("S_nom.jsonl", "S_nom_oracle.json"):
        (w8 / f).write_text("{}\n")
    p = _run(kit, "gpu-wave8-preflight.sbatch", e)
    assert p.returncode == 0, p.stdout + p.stderr
    text = calls.read_text()
    assert "wave8_grid.py --tokenizer-check" in text and "wave8_gates.py control" in text
    assert "--gates results/wave8/gates.json" in text


def test_all_new_sbatch_headers_follow_kit_pins():
    for name, part in (("cpu-wave8-label.sbatch", "tue.cpu3.q"), ("gpu-wave8-preflight.sbatch", "tue.gpu.q"),
                       ("gpu-wave8-train.sbatch", "tue.gpu.q"), ("gpu-wave8-eval.sbatch", "tue.gpu.q")):
        txt = (KITSRC / name).read_text()
        assert f"#SBATCH --partition={part}" in txt, name
        if part == "tue.gpu.q":
            assert "#SBATCH --constraint=l4" in txt and "#SBATCH --gres=gpu:1" in txt
        assert "EXPECTED_COMMIT" in txt and "HF_HUB_OFFLINE=1" in txt
    train = (KITSRC / "gpu-wave8-train.sbatch").read_text()
    assert "#SBATCH --signal=B:TERM@600" in train and "#SBATCH --array=0-5" in train
    assert "#SBATCH --time=08:00:00" in train                    # = config.sh WALLTIME_W8_TRAIN
    assert 'WALLTIME_W8_TRAIN="08:00:00"' in (KITSRC / "config.sh").read_text()
