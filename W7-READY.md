# W7-READY: independent verification of the AUDIT-2 fix bundle

**Overall sign-off**
- Item 1 (audit + fix history: 12 commits, every finding mapped): **PASS**
- Item 2 (hub suite: 326 passed / 15 skipped) **PASS**. Item 2b (exact-stack chunked): **SUPERSEDED** by live cluster evidence (job 390955, see Item 2b)
- Item 3 (P1-3 / P1-4 diff review against the audit): **PASS**. Item 4 (contract-test non-vacuity: 3/3 reverts caught): **PASS**
- Residual risks: **8** (top ones: the label hash conflict `59a6b39d…` vs `45d9f40d…`; L4 timing and memory `[UNVERIFIED]`); 1 new low-severity finding (R1), not fixed
- No code changed and no commits. The only tree change is this report (`W7-READY.md`, untracked).
- **SAFE TO SUBMIT: MOOT — already submitted and 2/3 seeds green on this revision.**

Repo `agent-bullwhip-replication` @ `2a8ff87` (master). Mission `/home/pino/missions/w7-ready`
(raw logs in `logs/`). Written item by item; each section is appended once its evidence exists.

## Item 1: audit and fix history

`WAVE7-AUDIT-2.md` read in full: P0-1, P1-1..5, P2-1..4, P3-1..11, and the 10-point ordered fix list.
`git log --oneline 8e53cbd..HEAD` has 12 commits. Each audit finding maps to one:

| finding | commit |
|---|---|
| P0-1 / A1 label mass (record, don't assert, for adapters; `min_label_mass` per SFT step) | `3854df1` |
| P1-2 RLCD exit reaches Slurm | `45b01c3` |
| P1-1 `main()` returns 75 unless `step >= steps-1` | `66dc51f` |
| P1-3 one adapter only (`_attach_adapter`) | `029e0f8` |
| P1-4 frozen π_sft → `pi_sft.json` + `labels_sha` | `f7005f2` |
| P1-5 skip round 2 when the manifest exists; adds `tests/test_sbatch_contract.py` | `8094d48` |
| P2-1 `verify_pull` 159 → 120 | `fa0184f` |
| P2-2 chunked `evaluate()` under `no_grad` | `0724ea4` |
| P2-3 full-file sha256 + gate / P2-4 refuse resume of an `aborted` state | `815ebda` |
| P3-5/6/7/11 (vset hoist, RLCD per-step reseed, sha `[:12]` guard, comment) | `4ac4555` |
| P3-1..4, 8..11 (SFT_FINAL from state.json, pipefail, empty-array, sanity line, whole-script trap, …) | `2a8ff87` |

Untracked files in the tree (not mine, left alone): `PROGRESS.md`, `PROGRESS_DESKTOP.md`,
`noisy_eval_arms.sh`.

## Item 2: test suites

**Hub venv** (`.venv/bin/python -m pytest tests/ -q`, log `logs/1791266535-hub-pytest.log`):

```
...................................................                      [100%]
326 passed, 15 skipped in 24.79s
```

**Exact cluster stack** (torch 2.1.2+cpu / transformers 4.55.4 / peft 0.21.1, `HF_HUB_OFFLINE=1`): the
monolithic run gets SIGKILLed on this 15 GB hub, so it was run chunked via `run_suite_chunked.sh`, one
test file per process. See Item 2b.

### Item 2b: exact-stack suite, SUPERSEDED

**SUPERSEDED.** The final revision `2a8ff87` was run end-to-end on the real cluster exact stack (torch
2.1.2 + CUDA, L4). In job **390955**, seeds **0 and 2 COMPLETED**, with SFT → collect → relabel →
RLCD all `exit=0` (operator-reported; I did not read that job's log from this hub).

The local chunked rerun was killed by the host repeatedly, so I relied on the cluster run instead.
Three attempts:
- Each passed the files it reached: `test_chaos_demand` 3, `test_cli_entrypoints` 4,
  `test_configurator` 4, `test_decision_labeler` 25, `test_decision_readout` 16.
- Each died on entering `test_decision_train.py`, taking the whole driver with it (SIGTERM/SIGKILL
  under host load on this 15 GB box). Logs: `/tmp/w7-chunked.run1-orphaned.log`,
  `logs/1791271240-w7s-chunked2.log`.
- The third run was stopped on the operator's instruction.

Gaps the live run does not close:
- **Seed 1 status: not reported here.**
- The torch-only unit tests after `test_decision_readout` (`test_decision_train`,
  `test_wave7_adapters`, …) never ran on the exact stack in this verification.
- A first-attempt cluster job exercises only the RLCD **fresh-start** path (single adapter from the
  SFT checkpoint, π_sft computed). The **resume** paths for P1-3 and P1-4 (load the RLCD checkpoint
  alone; load `pi_sft.json` or refuse) are verified only by diff review (Item 3).
  `[UNVERIFIED on the exact stack]`

## Item 3: diff review of P1-3 (`029e0f8`) and P1-4 (`f7005f2`), PASS

### P1-3: no PeftModel-on-PeftModel stacking on an RLCD resume. PASS
- Old code (`decision_train.py`, pre-fix): `if phase=="rlcd": from_pretrained(base, base_adapter)`, then
  **also** `if adapter_dir: from_pretrained(<that PeftModel>, adapter_dir)`. That is the audit's stack.
- New `_attach_adapter(model, phase, base_adapter, adapter_dir)` is a strict if/elif/else that returns
  at the first branch: `adapter_dir` → one `from_pretrained(base, adapter_dir)`; else `rlcd` →
  `from_pretrained(base, base_adapter)`; else a fresh `get_peft_model(LoraConfig(r=8, α=16, dropout
  0.05, q/k/v/o))`. No path can wrap twice. This matches the audit's prescribed fix word for word.
- **The SFT resume path is unchanged.** Before: `phase=="sft"` with `adapter_dir` → a single
  `from_pretrained(base, adapter_dir)`, with the `elif sft` fresh-LoRA branch skipped. After: the same
  single call. The fresh SFT path keeps the same LoraConfig values. The `requires_grad` loop that
  follows ("peft loads from_pretrained adapters FROZEN") is untouched, so the resumed and RLCD
  adapters stay trainable.
- `main()` sets `adapter_dir` only when `--resume` finds a `state.json` with a `checkpoint_dir`. So a
  fresh RLCD start = SFT adapter, and an RLCD resume = the RLCD checkpoint alone (which already holds
  the SFT-initialised weights, trained in place).
- New test `tests/test_wave7_adapters.py::test_rlcd_resume_loads_the_checkpoint_only` uses a tiny
  from-config Qwen3 and asserts four things: the fresh RLCD start carries the SFT LoRA exactly; the
  resume is not `PeftModel→LoraModel→PeftModel`; LoRA tensors and logits match the saved RLCD
  checkpoint (`atol 1e-6`) with `lora_B ≠ 0`; a re-save keeps the flat `base_model.model.model.` key
  prefix (the audit's "later checkpoints load as base" symptom). It runs only on torch hosts, i.e.
  in the exact-stack suite.

### P1-4: the KL reference is frozen, persisted, and gated on resume. PASS
- **Computed once.** `main()` calls `prepare_pi_sft(...)` only for `phase=="rlcd" and beta_kl>0`,
  after `_Policy` is built and the optimizer loaded, and before `run_loop` (so before step 0).
  `rlcd_step` no longer fills any cache: `_sft_ref_logprobs` is a pure lookup and raises
  `RuntimeError` if the table was never set.
- **Eval mode, `no_grad`, batched.** `compute_pi_sft` runs `model.eval()` and `torch.no_grad()`,
  chunks `rows` by `max(1, batch)` (= `--batch 32`, so 5,440/32 ≈ 170 forwards), and restores
  `model.train()` in a `finally`. `_Policy.__init__` already put the model in train mode (`:547`), so
  the restore returns it to its prior state. With grad-ckpt, transformers only checkpoints in
  training mode, so the eval-mode no-grad pass does not need checkpointing.
- **Keyed by row identity, not `id()`.** `pi_sft_key = f"{episode}:{week}:{role}"`. I checked that
  keys are unique in the real round-2 data: `collect_policy_states` uses `ep in range(n_episodes)` (a
  global index, not per curriculum), each episode's `week` is distinct, and roles are distinct per
  week. The relabel copies `episode/week/role` through unchanged. Duplicates raise `ValueError`
  rather than overwrite.
- **Prompt alignment.** Both `compute_pi_sft` and `rlcd_step` render `render_prompt(r["ctx"], tok)`
  with no option permutation, so the reference distribution is over the same label order as the
  policy's.
- **Persisted with the sha.** `_atomic_write_json(outdir/"pi_sft.json", {"labels_sha", "n", "pi"})`,
  plus a `{"event":"pi_sft","source":"computed"|"loaded"}` line in `train_log.jsonl`.
- **Resume refuses** (`SystemExit`, so non-zero) in three cases: the file is missing; the file's
  `labels_sha` ≠ the current `labels_sha` (full-file sha256 from `load_labels`); or any current row is
  absent from the table. It never rebuilds from the live adapter.
- **Edge cases checked.**
  - Crash after `pi_sft.json` is written but before the first `state.json`: resubmit → `payload is
    None` → `resuming=False` → recomputed from the freshly loaded SFT adapter. Correct, because the
    policy is still π_sft.
  - A pre-fix RLCD dir (`state.json` but no `pi_sft.json`) refuses loudly; it does not silently
    proceed.
  - `--beta-kl 0`: no table is built, and `rlcd_step` skips the KL branch, so it never calls the
    lookup.
- The step-0 KL is no longer ≈ 0 by construction: the reference is computed in eval mode and the
  policy trains in train mode with dropout, and the gap grows as RLCD moves θ. The first-run watch
  list (below) says what to check.

## Item 4: `tests/test_sbatch_contract.py` non-vacuity, PASS

The test runs the **real sbatch file** under bash with a fake `python3`: the real
`decision_train.main()`/`run_loop`/`save_checkpoint` with a stubbed `_Policy`, plus recorded fakes for
collect and relabel. The probe used four `git clone`s of HEAD under `/tmp/w7ready-probe/` (the working
tree was not touched). Three of them each revert exactly one fix; `base` is unmodified.

| clone | mutation | contract result | the failing test and its observed text |
|---|---|---|---|
| `base` | none | **5 passed** | n/a |
| `p12` | P1-2: delete `if [ "${RC}" -ne 0 ]; then exit "${RC}"; fi` after the RLCD wait loop | 1 failed, 4 passed | `test_rlcd_failure_exits_nonzero - AssertionError: rc=0`; stdout shows `=== rlcd seed=0 exit=1 ===` then `=== wave7-train seed=0 done ===` (the audit's masked exit, reproduced) |
| `p15` | P1-5: `if [ -s "${POLICY_LABELS}.manifest.json" ]` → `if false` | 1 failed, 4 passed | `test_round2_skipped_when_manifest_exists - AssertionError: ['train-sft', 'collect', 'relabel', 'train-rlcd']` |
| `p11` | P1-1: `main()` `return 75` → `return 0` | 1 failed, 4 passed | `test_sigtermed_sft_exits_nonzero_before_round2 - AssertionError: rc=0`; stdout: `SIGTERM from Slurm -> stage=sft` … `sft seed=0 exit=0` → `sft final adapter: …/checkpoint-4` → `rlcd seed=0 exit=0` → `done` (probe D1, reproduced) |

Each revert breaks exactly the test that names it, and the other four still pass. So the assertions
are specific, not vacuous. The same probe also confirms that the sbatch gates work as the audit claims:
- P1-2: an RLCD failure gives a non-zero script exit.
- P1-5: an existing manifest skips collect and relabel.
- P1-1: a SIGTERM'd SFT exits non-zero (75 from `main()`) before round 2, with `state.json` at
  `3 <= step < 199`.

## Additional finding R1 (low severity; not fixed, per the operator's "do not modify source files")

**The stage `wait` idiom can mask an instantly-failing child as `RC=0`.** Both stages use
`cmd & PY_PID=$!; RC=0; while kill -0 "$PY_PID"; do if wait …; then RC=0; else RC=$?; fi; done`.
If the child is already gone before the first `kill -0`, the loop body never runs and `RC` stays 0.
Probe `/tmp/w7race.sh` copies the sbatch lines verbatim:

| child | runs | masked as RC=0 |
|---|---|---|
| `/bin/false` | 500 | **4** |
| `/nonexistent/python3` (exec fails, exit 127) | 300 | **1** |
| `python -c 'raise SystemExit(1)'` (a real interpreter that starts, then fails) | 300 | 0 |

Impact on this job is small:
- Python takes long enough to start that only exec/loader-level failures can race.
- The sbatch calls `python3` earlier for the manifest hash, under `set -e`, so a missing interpreter
  dies before SFT.
- A masked SFT would still hit `FATAL: no sft checkpoint dirs`.
- A masked RLCD would need `python3` to vanish between stages.

It is still the same exit-code-masking class as P1-2. The fix is to reap once before looping, in both
stages:

```bash
RC=0; wait "${PY_PID}" || RC=$?
while kill -0 "${PY_PID}" 2>/dev/null; do RC=0; wait "${PY_PID}" || RC=$?; done
```

bash keeps the exit status of an already-reaped background PID, so the first `wait` always returns the
child's real code. After a trap-interrupted wait (143), the loop re-waits for python's own code (75).
Recommended for the next code change; it does not block this submission.

## Item 5a: resubmit commands (cluster login node)

Variable names are taken from `hpc-umbrella/gpu-wave7-train.sbatch` (`EXPECTED_COMMIT` `:61`,
`W7_EXPECT_LABEL_HASH` `:99`, `W7_EXPECT_LABEL_SHA256` `:108`) and `config.sh`
(`BULLWHIP_ROOT=/scratch-shared/$TUE_USER/bullwhip`, `REPO_DIR=$BULLWHIP_ROOT/agent-bullwhip-replication`,
kit at `$BULLWHIP_ROOT/kit`). All of them must be **exported**: Slurm's `--export=ALL` forwards only
exported variables (AUDIT-2 brief item 1).

**Staging trap (checked in `stage.sh:49-74`).** `stage.sh` mirrors **its own directory** into `kit/`
*before* it runs `git pull`. Two consequences:
- Run from inside `kit/`, it never refreshes the kit, so the submitted sbatch stays the old one.
- Run from `$REPO_DIR/hpc-umbrella` without pulling first, it copies the **pre-pull** sbatch.

So the order is: pull, then stage from the repo copy, then diff the kit against the repo.

```bash
B=/scratch-shared/$USER/bullwhip; R=$B/agent-bullwhip-replication
export EXPECTED_COMMIT=2a8ff87aa27ad171adfb364ce8c574decce71f3e   # = hub `git rev-parse HEAD` (pushed); do NOT compute it on the cluster after the pull (circular)

# 1. pull (ssh remote; stage.sh repeats this and gates on EXPECTED_COMMIT)
git -C "$R" fetch --all --prune && git -C "$R" checkout master && git -C "$R" pull --ff-only
test "$(git -C "$R" rev-parse HEAD)" = "$EXPECTED_COMMIT" || echo "STOP: cluster HEAD != $EXPECTED_COMMIT"

# 2. stage from the freshly pulled repo copy (mirrors hpc-umbrella/ -> $B/kit, re-execs, venv/weights idempotent)
bash "$R/hpc-umbrella/stage.sh"
diff -q "$B/kit/gpu-wave7-train.sbatch" "$R/hpc-umbrella/gpu-wave7-train.sbatch" && echo "kit sbatch == repo sbatch"
( cd "$B/kit" && bash config.sh --check )        # must end "RESULT: 0 fail / N warn"

# 3. pre-submit label check (see "label-hash gates" below)
python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["hash"])' "$R/results/wave7_labels/labels.manifest.json"
sha256sum "$R/results/wave7_labels/labels.jsonl"

# 4. stale-output check: no pre-fix round-2 / RLCD state may be reused (P1-5 skip + resume would pick it up)
ls -la "$R"/results/wave7_sft_seed* "$R"/results/wave7_rlcd_seed* "$R"/results/wave7_labels/policy_* 2>&1
#    anything listed here comes from a pre-fix attempt -> move it aside (mv … "$R/results/_prefix_$(date +%s)/") before submitting

# 5. submit
export W7_EXPECT_LABEL_HASH=59a6b39d93c62a933568f73c7202ef0c034addd77a24b5edd6eb8d87412e9aaa   # only after step 3 printed exactly this
export W7_EXPECT_LABEL_SHA256=<the sha256sum printed in step 3>                                # see below
cd "$B/kit" && sbatch gpu-wave7-train.sbatch
```

### Label-hash gates: what to set
- **`W7_EXPECT_LABEL_HASH`** compares against the manifest `hash`, which covers only the **first 2,000
  of ~95,200 rows** (`--hash-rows 2000`). **Discrepancy:** the brief gives `59a6b39d…`, but the sbatch
  comment (`:89`, added in `d92ec82` for this exact gate) names `45d9f40d6ffc4ce93b0caaf7e9233c1e22e6b726efc980d92ffd8dc13e41f51b`
  as the "canonical hub hash". No wave-7 label set exists on this hub (`results/wave7_labels/` is
  absent), and no log or mission file records either value, so I could not tell which one the
  cluster manifest holds. `[UNVERIFIED]`. Step 3 prints the cluster value.
  - If it equals `59a6b39d…`, export it as written.
  - If it equals `45d9f40d…`, the brief's value is stale. Export the printed value **only if** it also
    matches the hub reference used for the C4 determinism check (OPERATOR.md: "compare it with a hub
    run of the same knobs").
  - If it equals neither, **do not train** (OPERATOR.md failure playbook: "labels hash ≠ hub →
    different code/knobs").
  - If the wrong value is exported, the job fails cheaply at startup (`FATAL: label hash …`, exit 1,
    before SFT), so the gate is safe to set.
- **`W7_EXPECT_LABEL_SHA256`** (the full-file gate) **should be set.** It is the only gate covering all
  rows, and it equals the trainer's `labels_sha` (proven in `test_full_file_label_sha256_gate`). The
  hub has no copy of the cluster `labels.jsonl`, so the value cannot come from here. Set it to the
  `sha256sum` printed in step 3, **after** the 2000-row hash has matched. This does not re-prove
  determinism; the 2000-row match against the hub does that. What it buys:
  - all three array tasks verifiably train on one and the same file;
  - a later relabel or overwrite of `labels.jsonl` between resubmits is refused at startup instead of
    surfacing as a `labels hash changed` refusal deep inside the SFT resume.
  - Record the value in the submission notes. It must equal `labels_sha` in every
    `wave7_sft_seed*/train_log.jsonl`.

## Item 5b: watching the first run

Slurm log: `/scratch-shared/$USER/bullwhip/logs/wave7-train_<jobid>_<task>.log`. Per-step metrics are
**not** in the Slurm log; `_log` writes only to `results/wave7_{sft,rlcd}_seed<N>/train_log.jsonl`.

**Startup (first ~1 min).** Each of these must appear, in this order:
- `repo … HEAD=2a8ff87a… dirty_files=0`
- `labels: … hash(first 2000 rows)=…`
- `sha256(full file)=…`
- `python: …/venv/bin/python3 Python 3.11…`
- `torch 2.1.2 cuda True`
- `nvidia-smi` showing an `NVIDIA L4` (~22 GiB `memory.total`)

**Stop if:** `cuda False`, a FATAL, or a dirty tree.

**Stage exit codes.** Each line is `=== <stage> seed=N exit=RC ===`.

| RC | meaning | action |
|---|---|---|
| 0 | stage done | none |
| 75 | SIGTERM'd/partial (P1-1) | resubmit the same array task; `--resume` continues |
| 1 | refusal or Python exception | read the traceback; **do not** blind-resubmit. Refusals: `labels hash changed`, `pi_sft.json missing/mismatch`, `run aborted`, `resume phase mismatch` |
| 3 | non-finite abort (`aborted: nonfinite`); P2-4 refuses to resume it | inspect `checkpoint-prenan-*` |
| 137 | host cgroup OOM (`--mem=16G`) | — |
| CUDA OOM | traceback in the log | — |

On success the log shows `round 2: … manifest exists — reusing` only on a resubmit. On a first run it
shows collect and relabel output.

**A1: `min_label_mass` curve** (SFT `train_log.jsonl`, every step; `mean_label_mass` next to it):
```bash
python3 -c 'import json,sys;[print(r["step"],r.get("min_label_mass"),r.get("mean_label_mass")) for r in map(json.loads,open(sys.argv[1])) if "step" in r and "event" not in r]' results/wave7_sft_seed0/train_log.jsonl
```
- Expected: it starts around 0.998 (the base had min 0.99835) and drifts down. The audit's real-lr
  proxy extrapolated a crossing of 0.99 near step 40/120, so dropping below 0.99 is **expected and
  no longer fatal** (A1).
- The round-2 collect writes `results/wave7_labels/policy_states_seed<N>.jsonl.notes.json` with
  `min_mass` and `median_mass` over all ~5,760 visited states.
- **Stop and escalate to the owner if** `median_mass` < ~0.5, or the SFT `mean_label_mass` keeps
  falling steeply through step 119. The 21-letter argmax stays well defined, but by then most
  probability sits outside the option letters, which is a format collapse. This threshold is my
  judgment, not a spec value.

**Per-stage timings vs the audit estimates** (AUDIT-2 walltime section; all `[UNVERIFIED on L4]`):

| stage | estimate | where to read it |
|---|---|---|
| SFT, 120 steps | 2–4 min (≈1–1.6 s/step) | `wall` field per step in SFT train_log; the `=== sft … at <date>` stamp vs the start stamp |
| collect, 40 episodes | 1–2 min | gap from `sft final adapter` to the relabel start |
| relabel, ~5,440 states / 8 processes | 4–10 min | until the RLCD start |
| RLCD | `pi_sft` precompute (~170 no-grad batches) <1 min + 40 steps 1–2 min | `{"event":"pi_sft","source":"computed","n":…}` then the `wall` field |
| total | 15–30 min against the 4 h limit | — |

If SFT runs > ~10 s/step (> 20 min for SFT), it is far off the estimate. That is not fatal (4 h
covers it), but record it before scaling `W7_SFT_STEPS`. If any task nears 3 h 50 min, `TERM@600`
fires → exit 75 → resubmit.

**First L4 memory numbers.**
- The job logs **no GPU peak memory**: there is no `torch.cuda.max_memory_allocated` call anywhere in
  `agent_bullwhip/`, and only `memory.total` is printed.
- While a task runs: `srun --jobid=<jobid>_<task> --overlap nvidia-smi --query-gpu=memory.used --format=csv`.
- After it finishes: `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS` (host RSS vs
  `--mem=16G`).
- Audit expectation with grad-ckpt: ≈4–4.5 GiB realistic, ≤7.5 GiB upper bound, on 22 GiB.
- **Stop if** usage exceeds ~15 GiB during SFT: the probe B model would then be wrong, and batch 32
  has little margin.

**RLCD sanity** (`wave7_rlcd_seed<N>/train_log.jsonl`):
- `pi_sft` event with `source: computed` and `n` = the number of round-2 rows.
- Step-0 `loss_kl`: small but no longer pinned at the audit's `1e-6`/`-0.0`, and growing over the 40
  steps.
- `pi_sft.json` present in the RLCD dir.
- A flat `loss_kl ≈ 1e-6` across all steps would mean P1-4 is not active → stop.

**After all three tasks finish:** pull with `pull_results.sh`, then run `verify_pull.py`. With P2-1
fixed it should say COMPLETE: SFT dirs at step 119 / checkpoint 119, RLCD dirs at step 39.

## Residual risks

1. **Which label hash applies is unresolved.** The brief says `59a6b39d…`; the sbatch comment says
   `45d9f40d…`. `[UNVERIFIED]`: no hub label set exists to check against. The pre-submit step 3
   settles it on the cluster. A wrong export costs one fast FATAL, not a run.
2. **L4 behaviour is unproven:** walltime, GPU memory, bf16 numerics, and SDPA vs the math fallback
   with the padding mask. `[UNVERIFIED on L4]`. Every estimate here comes from AUDIT-2's CPU probes and
   FLOP arithmetic. The job logs no GPU peak memory (watch list above).
3. **Label-mass drift past A1.** A1 makes low mass non-fatal for adapters. It does not say how low is
   still a meaningful policy. A steep collapse is an owner decision, not something this job will
   catch. `[UNVERIFIED]`: the real step-120 value on the L4.
4. **bf16 round-2 determinism.** With P1-5 the round-2 bytes are produced once and reused, so
   reproducibility no longer matters *within* a campaign. Bit-reproducing them across campaigns on
   the L4 is `[UNVERIFIED]` (AUDIT-2 P1-5).
5. **Stale cluster outputs.** The cluster state of `results/wave7_{sft,rlcd}_seed*` and
   `wave7_labels/policy_*` is `[UNVERIFIED]` from here. The audit's history says earlier attempts died
   in SFT (OOM at step 0), so round-2 files are not expected. If a pre-fix `policy_labels_seed<N>.manifest.json`
   exists, P1-5 would *skip* round 2 and train RLCD on it. Pre-submit step 4 covers this. Pre-fix
   RLCD dirs without `pi_sft.json` fail loudly (P1-4 refusal), not silently.
6. **R1: the wait-loop race** (above). Low probability, and only for exec/loader-level failures.
   Not fixed.
7. **`verify_pull.py` RLCD expectation `39`** contradicts its own "value = total steps" comment
   (RLCD = 40 steps). It is lax, not wrong for a real success: it would also pass an RLCD whose last
   logged step is 38. P1-1's exit 75 already keeps a partial RLCD from reporting success. Hygiene only.
8. **The P1-3 test runs on torch hosts only.** `tests/test_wave7_adapters.py` skips in the hub venv
   (torch absent), and none of the three exact-stack chunked attempts reached it (Item 2b). The RLCD
   *resume* paths of P1-3 and P1-4 have no executed evidence in this verification; job 390955 covers
   the fresh-start path only. `[UNVERIFIED]`
