# WAVE7-AUDIT-2: run-time defects in the wave-7 training pipeline

Scope: `hpc-umbrella/gpu-wave7-train.sbatch` (array 0–2, 8 CPU, 1×L4 22 GiB, 4 h, `--signal=B:TERM@600`,
HF offline), in the order a job runs it. Audited at `9a0abcf`. `8e53cbd` (a stage.sh hardening commit
that touches only `stage.sh`) landed during the audit; no pipeline code changed. Design is taken as
frozen (AUDIT-1 / WAVE7_SPEC). The only question asked here: what kills or silently corrupts the next
cluster job.

Probes are in `/tmp/w7audit2/` and ran CPU-only on the cluster's exact stack (torch 2.1.2+cpu,
transformers 4.55.4, peft 0.21.1, pinned Qwen3-0.6B snapshot `c1899de`, `HF_HUB_OFFLINE=1`). Raw logs are
in `/home/pino/missions/w7-audit-2/logs/`.

| probe | what it does |
|---|---|
| A `e2e.sh` | The sbatch's stage chain at reduced size, using the real CLIs and the sbatch's flags: label → SFT (`--grad-ckpt --resume`) → verbatim `SFT_FINAL` line → `decision_eval --adapter … --collect-states` ×2 → `decision_labeler --states-jsonl` ×2 → RLCD → RLCD requeue → RLCD requeue on changed labels |
| B `memprobe.py` | Bytes autograd retains in one SFT forward (saved-tensor hooks, parameters excluded), with and without grad-ckpt, using real prompts; scaled to B=32 |
| D `sim/run_d.sh` | Runs the **actual sbatch file** under bash with a fake `config.sh`/`python3`. `decision_train.main()`, `run_loop` and `save_checkpoint` are the real code; only `_Policy` is stubbed. Then sends SIGTERM and injects failures |
| E `nested_peft.py`, `nested_peft2.py` | Checks what an RLCD resume loads, and whether checkpoints saved after it load in `decision_eval` |
| F `valsplit_cost.py` | Train/val split cost at 95,200 rows; whether validation ever fires |
| G (inline) | `verify_pull.py` run on a synthetic pull of a fully successful default run |
| H (inline) | Edge cases of the `SFT_FINAL` glob/sort |
| I `mass_drift.py` | Real `_Policy.sft_step` on real labels; tracks the full-vocab mass on the 21 option letters (the quantity `readout_batch` asserts ≥ 0.99) |

---

## P0: blocks the run

### P0-1 SFT erodes the option-letter mass and round-2 collect asserts on it, killing every array task right after SFT
- **Evidence.** `decision_eval.py:151` calls `model.readout_batch(prompts)` with the default
  `require_mass=0.99`, and `decision_readout.py:297,328-331` raises `AssertionError` on any row below
  that. The SFT loss (`decision_train.py:564-567`) is a softmax over the 21 label logits only. No term
  holds up the full-vocab mass on those letters.
- **Probe I.** The untrained base starts with only 0.008 of margin (min 0.99835 over 32 states). At the
  real lr 1e-5 (batch 4), min mass drops to **0.99801 after 5 steps and 0.99622 after 10**, and the decline
  is speeding up. A stress run that matches Adam's displacement for 120 steps at 1e-5 (10 steps at 1.2e-4)
  ends at **min 0.92686**. A linear extrapolation of the real-lr curve crosses 0.99 around step 40 of 120.
  The collect then visits 5,760 states, not 32, so the minimum will only be lower.
  `[PROXY]`: 10 of 120 steps, batch 4, CPU fp32; the exact value at step 120 on the L4 was not run.
- **Effect.** Each task gets through SFT (~5 min), dies in week 0 of the collect with exit 1, and leaves
  no round-2 data and no RLCD. The same assertion would also stop every trained-adapter eval in
  OPERATOR step 4.
- **Fix (owner decision; WAVE7_SPEC:54/205/253 makes "mass ≥ 0.99 fatal" a pre-registered invariant).**
  Pick one:
  - (a) Narrow the invariant. Keep it fatal for the untrained readout, where it is the format proof.
    For adapters, record per-row mass instead of asserting. The argmax over the 21 letters is well
    defined either way. For this job only, the minimal unblock is `readout_batch(prompts,
    require_mass=0.0)` in `collect_policy_states` plus writing `mass` into each state row. The relabel
    never reads mass.
  - (b) Add a mass-retention term to the SFT loss. That is a design amendment.

  Either way, log `min_label_mass` on every SFT step. `_label_logits` already has the full-vocab
  `last` row, so `(cand.logsumexp - last.logsumexp).exp().min()` costs nothing.
- **Verified.** Probe I (two runs, logs `*mass-drift*.log`); code path read.

No other P0 was found on the happy path. Env and entry, the SFT memory, the adapter handoff, collect
loading the peft directory, relabel schema compatibility, and the RLCD first start all ran clean in
probe A.

---

## P1: silent corruption, or costs a cluster cycle

### P1-1 A SIGTERM'd SFT exits 0, so the job runs round-2 and RLCD on a half-trained adapter
- **Evidence.**
  - `decision_train.py:283-291`: on SIGTERM it saves, logs `sigterm` and `return`s. `main()` then
    returns 0 (`:389`).
  - The sbatch (`:117-125`) only stops on `RC != 0`, so `SFT_FINAL` (`:130`) resolves to the partial
    checkpoint and collect, relabel and RLCD all run on it.
  - The same applies to a SIGTERM'd RLCD: the job reports COMPLETED.
- **Probe D1.** SIGTERM after 6 s: `sft exit=0`, `sft final adapter: …/checkpoint-13`, collect from
  checkpoint-13, RLCD trained to completion, `sbatch script exit status: 0`.
- **Probe D3.** Resubmitting that task: SFT finishes (checkpoint-39), the relabel produces new bytes,
  and RLCD refuses (`labels hash changed`) with exit 1. The job still exits 0, and
  `results/wave7_rlcd_seed0/` keeps the RLCD trained from the partial SFT.
- **Fix.** At the end of `main()`:
  `st = load_resume_state(outdir)[1]; return 0 if st and st["step"] >= args.steps - 1 else 75`.
  The sbatch already exits on non-zero.
- **Verified.** Probe D1/D3 on the real sbatch file and the real `main`/`run_loop`.

### P1-2 RLCD failures never reach Slurm: the job is COMPLETED and the mail says END
- **Evidence.** `gpu-wave7-train.sbatch:174-176` prints `rlcd … exit=${RC}` and then `done`. There is no
  `exit`, so the script's status is that of the last `echo`.
- **Probe D2.** RLCD raising at model load gives `rlcd seed=0 exit=1` and `sbatch script exit status: 0`.
- **Fix.** Before the `done` line, add `if [ "${RC}" -ne 0 ]; then exit "${RC}"; fi`. Use the `if` form:
  a trailing `[ … ] && exit` would itself become the script's status 1 on success.
- **Verified.** Probe D2.

### P1-3 An RLCD resume stacks PeftModel on PeftModel: both adapters are wiped to the base model, and later checkpoints load as the base in eval
- **Evidence.** `decision_train.py:452-455`. With `--phase rlcd --resume` and an existing state, it runs
  `PeftModel.from_pretrained(PeftModel.from_pretrained(base, SFT_FINAL), rlcd_ckpt)`. peft warns
  "Already found a `peft_config`…" and then "Found missing adapter keys": the checkpoint keys are
  `base_model.model.model.layers…` but the nested model expects `base_model.model.base_model.model…`.
- **Probe E/E2.**
  - The nesting is `PeftModelForCausalLM → LoraModel → PeftModelForCausalLM`, with one `default` adapter
    (224 LoRA tensors). The outer injection re-initialises the shared `default` adapter: **all 112
    `lora_B` tensors are exactly 0**, so the resumed policy is the base model. SFT and RLCD weights are
    both gone.
  - Readout distance: `max|p_resume − p_rlcd_ckpt| = 0.137`, `|p_resume − p_sft| = 0.025`.
  - The old optimizer state loads without error onto the fresh adapter (224/224 params).
  - Checkpoints saved after the resume carry the nested key prefix. `decision_eval --adapter` on them
    gives a warning only and **`max|p − base| = 1.8e-6`**: the "trained RLCD" eval is the untrained base.
    `verify_pull` still reports payload `yes`.
- **Probe A step 7.** The resume ran with `rc=0` and gave no error.
- **Fix.**
  ```python
  if adapter_dir is not None:  self.model = PeftModel.from_pretrained(self.model, adapter_dir)
  elif phase == "rlcd":        self.model = PeftModel.from_pretrained(self.model, base_adapter)
  else:                        self.model = get_peft_model(self.model, cfg)
  ```
  The RLCD checkpoint already holds the SFT-initialised adapter, which RLCD trains in place.
  E2 confirms `PeftModel(base, rlcd/checkpoint-1)` loads with no missing keys and carries the training.
  This needs P1-4's fix for the reference distribution. SFT resume is not affected, since it is a single
  `from_pretrained` (`55bf75a` roundtrip).
- **Verified.** Probes E and E2, using the exact `_Policy` call that `main()` makes.

### P1-4 The KL anchor is inert: π_sft is recomputed from the policy being trained
- **Evidence.**
  - `_sft_ref_logprobs` (`decision_train.py:592-602`) fills its cache lazily from `self.model`, which is
    the live adapter under training, in train mode with dropout on.
  - `rlcd_step` (`:610-611,635-640`) asks for the reference of each batch just before that batch's
    update.
  - 40 steps × 32 = 1,280 rows, drawn without replacement from 5,168 training rows (probe F), so every row
    is first seen in its own step. Its reference is then π_θ at that step, and KL ≈ dropout noise.
  - The spec requires "pi_sft frozen per state (computed once from the SFT adapter)" (docstring `:16-17`,
    SPEC:214/256).
- **Probe A.** `loss_kl` = `1e-06`, `1e-06`, `-0.0` over the three RLCD steps, so `--beta-kl 0.02` does
  nothing. On resume the cache is empty and the reference is rebuilt from the resumed adapter (after the
  P1-3 fix, from the RLCD weights), so resume is not exact either.
- **Fix.** On a fresh RLCD start, before step 0, compute π_sft for every row: eval mode, `no_grad`,
  batched. Key it by `(episode, week, role)`, not `id()`. Write it with `_atomic_write_json` to
  `outdir/pi_sft.json` together with `labels_sha`. On resume, load that file and refuse if it is missing
  or the sha differs. Cost: about 170 no-grad batches, under 1 min on an L4.
- **Verified.** Code trace plus the probe A train_log values.

### P1-5 Round 2 is re-run on every resubmit and is not idempotent: RLCD either refuses or goes down P1-3
- **Evidence.** `gpu-wave7-train.sbatch:138-152` always re-runs collect and relabel. Collect overwrites
  in place (`decision_eval.py:138`). The labels_sha gate (`decision_train.py:351-356`) compares the new
  relabel bytes with the RLCD state.json. Two outcomes:
  - The bytes differ → refusal, exit 1, masked by P1-2 (probe A step 8: `rlcd-changed-labels rc=1`).
  - The bytes match → the resume proceeds into P1-3 and silently wipes the adapter.

  Collect and relabel are byte-deterministic on CPU (probe A: identical sha256 for states ×2 and labels
  ×2; relabel `hash` identical). Bit-reproducibility of a bf16 argmax rollout on the L4 is
  `[UNVERIFIED]`, but same-architecture cuBLAS is normally deterministic, which points at the silent
  branch.
- **Answer to the brief's labels_sha question.** On the **first** attempt the gate does **not** refuse.
  RLCD writes to its own `wave7_rlcd_seed<N>/`, `load_resume_state` returns `(0, None)`, and the SFT's
  seed-6000 `labels_sha` is never compared with the round-2 set (probe A step 6, `rc=0`). The gate only
  matters on a resubmit, as above.
- **Fix.** Wrap collect and relabel in `if [ ! -s "${POLICY_LABELS}.manifest.json" ]; then …; fi`. The
  relabel writes the manifest last, and a killed collect exits non-zero before the relabel starts.
  P1-1's fix is required first, so that no manifest can come from a partial SFT.
- **Verified.** Probe A steps 4, 5 and 8.

---

## P2: breaks a reuse or a claim

### P2-1 `verify_pull.py` reports a fully successful run as INCOMPLETE
- **Evidence.** `verify_pull.py:21-23` expects `wave7_sft_seed*` to reach 159, read as "SFT + RLCD". But
  RLCD writes to `wave7_rlcd_seed*`, so the SFT dir ends at step 119.
- **Probe G.** A synthetic default-success pull gives `VERDICT: INCOMPLETE`, with `train_log last step
  119 < 158`, `state step 119` and `last checkpoint 119` for all three SFT dirs.
- **Fix.** Set those entries to `120` and fix the comment.

### P2-2 `evaluate()` holds full activations for the whole validation set in one forward, and the SFT never validates at all
- **Evidence.** `decision_train.py:652-671` runs no `torch.no_grad()` and sends all `val_rows` in one
  batch. `model.eval()` also switches off gradient checkpointing, because transformers checkpoints only
  when `self.training`.
- **Probe F.**
  - At the default knobs validation never fires. SFT has 2,827 batches per epoch against 120 steps;
    RLCD has 162 against 40.
  - So **no val nll/top1/regret is ever logged**, which breaks the "validation at each epoch boundary"
    claim (`:292`).
  - Validation does fire if `W7_N_COLLECT ≤ 9` (decision_eval's own default is 8): a 55-row
    single-batch forward with a graph and no checkpointing, about 1.7× the batch-32 activations that
    already OOM'd.
- **Fix.** Wrap the body in `torch.no_grad()` and chunk by `args.batch`. If val metrics are wanted, also
  validate at the last step.

### P2-3 The label-hash gate covers 2.1% of the label set
- **Evidence.** The sbatch (`:89-96`) compares the manifest `hash`, which is `_row_hash(rows, 2000)`
  (`decision_labeler.py:337-341,528-530`): the first 2,000 of 95,200 rows. A difference after row 2,000
  passes `W7_EXPECT_LABEL_HASH`. The full-file sha only appears inside `train_log.jsonl` (`labels_sha`).
- **Fix.** Also print `sha256sum "${LABELS_JSONL}"`. This equals the trainer's `labels_sha`. Gate on
  it, or rename the gate to say "first 2000 rows".

### P2-4 A non-finite abort is silently resumed on resubmit
- **Evidence.** `decision_train.py:270-282` writes `state.json` with `aborted: "nonfinite"` pointing at
  `checkpoint-prenan-<k>`. `load_resume_state` (`:177-182`) ignores `aborted`, so a resubmit continues
  from the pre-NaN weights with the same seeded batches.
- **Fix.** In `main()`, `if payload and payload.get("aborted"): raise SystemExit(f"run aborted ({…}) — inspect before resuming")`.

---

## P3: hygiene

1. **`SFT_FINAL` glob is fragile** (sbatch `:130`).
   - What works: correct for the current relative `results/…` path and for 3-digit steps (probe H →
     `checkpoint-119`). `checkpoint-prenan-*` sorts as 0. A top-step `.tmp` cannot coexist with RC 0.
   - What breaks it (probe H): any hyphen in the path. An absolute `/scratch-shared/…` path picks
     `checkpoint-prenan-57`. A `checkpoint-119.tmp` beats `checkpoint-109`.
   - Fix: read the authoritative pointer,
     `python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_dir"])' "${SFT_OUT}/state.json"`.
2. **`REPO_DIRTY` can kill the job silently** (sbatch `:57`). The `git status … | wc -l` pipeline has no
   `|| true`, so under `pipefail` a failing `git status` exits with no message. This is class #3.
3. **Empty array under `set -u`** (sbatch `:111,162`). `"${GRAD_CKPT_FLAG[@]}"` with `W7_GRAD_CKPT=0`
   expands an empty array, which is an "unbound variable" error on bash < 4.4. `[UNVERIFIED]`: the
   cluster's bash version. Fix: `${GRAD_CKPT_FLAG[@]+"${GRAD_CKPT_FLAG[@]}"}`.
4. **A SIGTERM during round 2 kills bash with no output.** After `trap - TERM` (`:121`) the default action
   kills bash immediately; the orphaned collect/relabel runs until the hard kill. Fix: keep a logging
   trap for the whole script.
5. **No interpreter/CUDA sanity line in the train sbatch**, although the label sbatch has one (`:67`). Add
   `echo "python: $(command -v python3)"; python3 -c 'import torch;assert torch.cuda.is_available()'`
   before SFT.
6. **`run_loop` start-up cost** (`decision_train.py:229`). `i not in set(val_idx)` rebuilds the set for
   every row: **11.8 s** at 95k rows (probe F). Labels are also parsed twice (`:349` and `:226`). Fix:
   hoist `vset = set(val_idx)`.
7. **RLCD dropout is unseeded per step.** `rlcd_step` does not reseed torch, unlike `sft_step:546`, so
   LoRA dropout masks are not resume-exact.
8. **Labels-changed message can crash.** `decision_train.py:355`: `payload.get('labels_sha')[:12]` raises
   `TypeError` if an old state has no sha, which masks the intended message.
9. **stage.sh weights step.** `stage.sh:109` calls `snapshot_download` with no `revision`, and
   `bw_load_env` (`config.sh:197`) has already exported `HF_HUB_OFFLINE=1`. The "download" is therefore
   offline, and the failure message blames egress. `config.sh:48-51` claims stage.sh downloads the pinned
   revision. Harmless today, since the SFT loaded `c1899de`, so it is present.
10. **Label sbatch exit message.** `cpu-wave7-label.sbatch:77`: `labeler exit=$?` always prints 0 (set -e).
11. **Stale or unverified comments.** The comment at `decision_train.py:398` says "~700-token prompts";
    the measured range is 408–440. The sbatch comment at `:14` ("measured ~1-2 s/batch") was never
    measured on an L4; the first L4 step OOM'd.

---

## The brief's audit items, answered

1. **Env/entry.**
   - Slurm's default `--export=ALL` passes everything exported in the submitting shell to all three
     array tasks, including `EXPECTED_COMMIT`, `W7_*`, `W7_EXPECT_LABEL_HASH` and
     `PYTORCH_CUDA_ALLOC_CONF`. Plain `VAR=x` without `export` does not survive. `config.sh` keeps
     incoming values (`${EXPECTED_COMMIT:-}`), and sbatch `:76` only defaults
     `PYTORCH_CUDA_ALLOC_CONF`. `expandable_segments` exists in torch **2.1.2**
     (`c10/cuda/CUDACachingAllocator.cpp@v2.1.2:1095`), so it will not trip the "Unrecognized
     CachingAllocator option" check.
   - libpython: the venv was made with `python3 -m venv --system-site-packages` from the module's
     EasyBuild Python. Its `bin/python` is a symlink to that binary, which finds
     `libpython3.11.so.1.0` only through the module's `LD_LIBRARY_PATH` `[INFERENCE: no RPATH; matches
     the observation]`. On a login shell without `module load`, the venv python fails to load. Inside
     the job, `bw_load_env` loads Umbrella/2024 + PyTorch/2.1.2 **before** activating the venv, so it
     works.
   - Array tasks: each task runs the whole script and calls `bw_load_env` itself, after `module purge`.
     None of them relies on inherited module state.
   - Residual risk: if `module` were missing in a batch shell, `bw_load_env` only prints WARN. The
     first `python3` call (sbatch `:91`, inside `$(…)`) would then die with exit 127 and the loader
     error on stderr, which is loud enough. P3-5 makes it explicit.
2. **SFT stage.**
   - `--resume` on a fresh directory is fine (`(0, None)`; probe A step 2).
   - `save-every=10` writes checkpoints 9…119.
   - **Memory after grad-ckpt** (probe B, real prompts T=440, scaled to B=32):

     | | no grad-ckpt | grad-ckpt |
     |---|---|---|
     | retained activations, fp32-sized | 55 GiB | 1.6 GiB (T=436) to 1.8 GiB (T=500) |
     | one-layer recompute | n/a | 2.0–2.3 GiB |
     | peak upper bound, incl. bf16 weights and 2× recompute | n/a | 6.6–7.5 GiB |
     | realistic bf16 peak | ≈28–32 GiB (matches the observed OOM at ~21.5 GiB allocated) | ≈4–4.5 GiB |

     So **≥2.9× headroom on 22 GiB at the upper bound (≈5× realistic)**. Caveat: CPU sdpa stored no T×T tensors. If the L4 falls back to
     the math backend because of the padding mask, one layer's recompute adds ≲1 GiB transient. Still
     fits.
   - **Walltime.** My assumption is about 4.8e13 FLOP per SFT step (fwd + recompute + input-grad bwd,
     0.44 B non-embedding params × 14k tokens) at 30–50 TFLOPS effective on an L4, i.e. ≈1–1.6 s/step,
     or ≈2–4 min for 120 steps.
     - Collect: 1,440 batch-4 forwards ≈ 1–2 min.
     - Relabel: measured 0.34 s/state/process on the hub (probe A, M=16), so 5,440 states / 8 processes
       ≈ 4 min (≤ 10 min on a slower node).
     - RLCD: 40 steps ≈ 1–2 min.
     - Loads/splits: 3 model loads plus 2 × ~12 s split.

     Total ≈ 15–30 min against a 4 h limit. The wave-6 92 s/step figure was GRPO with 36-step
     autoregressive rollouts and does not transfer. The `TERM@600` path only fires if something hangs.
     `[UNVERIFIED on L4]`.
3. **SFT_FINAL handoff.** `save_checkpoint` writes `.tmp` and renames it, and writes `state.json` last.
   After RC 0 no partial directory can be the maximum, and 3-digit numeric sort is correct for the
   relative path (probe H). It is fragile (P3-1), and with P1-1 it picks a *complete but early*
   checkpoint.
4. **Round-2 collect.** `--adapter <checkpoint dir>` with `--model` loads the `save_pretrained` LoRA
   directory: tokenizer files and `optimizer.pt` sit alongside, which is harmless. Offline, peft only
   warns about the base config. The rows carry every field that `_label_collected_task` reads
   (`episode/week/role/curriculum/params/ctx/il_hist/state{t,tiers,inbound,prev_incoming,order_history}`).
   `tiers` order matches `TierState(oh,b,o)`. The rollout is byte-deterministic for a given
   `--collect-seed` on CPU (probe A). It then dies on P0-1.
5. **Relabel.** `t ≥ H−2` exclusion: 8 of 144 rows are excluded per episode (probe A,
   `n_excluded_late=8`). `--episodes` and `--curricula` are unused in this mode (the curriculum comes from
   each row), so their absence is correct. M defaults to 16. Seed 6000+N does not collide with the eval
   band, and the RNG namespace `…:states:…` is disjoint from seed-6000's `…:ep:…:w:…`. Output is
   deterministic with `--processes 4` (identical sha256) and carries everything RLCD and `evaluate`
   read (`role, week, ctx, p, J, argmin_idx`). The hash covers only 2,000 rows (P2-3).
6. **RLCD.** The fresh start composes correctly: a single PeftModel over the SFT adapter with its LoRA
   made trainable. With grad-ckpt and `--beta-kl 0.02` it runs, rc 0 (probe A). The labels_sha gate
   does not refuse on the first attempt (P1-5). The defects are the inert KL (P1-4) and the resume
   stack (P1-3).
7. **SIGTERM/requeue.**
   - SFT resume is coherent: state written last, optimizer pre-rename, step-derived batches and
     dropout.
   - The bash trap forwards correctly (probe D1).
   - Incoherent on the RLCD side: exit 0 after SIGTERM (P1-1), adapter wipe on resume (P1-3), reference
     rebuilt (P1-4), round 2 re-run (P1-5), aborted state ignored (P2-4).
8. **Known bug class.**
   - `pipefail` silent exit at `:57` (P3-2).
   - Empty array under `set -u` (P3-3).
   - Exit-code masking (P1-2) and a 0 exit on an incomplete run (P1-1).
   - Silent death after the trap is reset (P3-4).
   - No argparse mismatches: every sbatch flag exists in its parser. Checked: `decision_train` (all 14
     flags), `decision_eval` (`--model --adapter --collect-states --n-collect --collect-seed`),
     `decision_labeler` (`--states-jsonl --seed --processes --hash-rows --out`).
   - CUDA OOM tracebacks reach the log, because stderr goes to `--output`. A host cgroup OOM shows as
     `exit=137`.
   - Array-only failure: `${SLURM_ARRAY_TASK_ID}`/`${SLURM_CPUS_PER_TASK}` under `set -u` crash any
     non-array or manual run. Intended, not a defect.

---

## What breaks next

With grad-ckpt the next submission should get through SFT quickly: about 4 GiB peak, a few minutes for
120 steps. Then all three tasks will die in the first week of the round-2 collect on the `label mass
< 0.99` assertion, because nothing in the SFT loss holds the option-letter mass up. It starts 0.008 above
the threshold, and at the real lr it already falls ~0.002 per 10 steps. That will look like a fourth
"found only on the cluster" failure, at roughly the 5–10 minute mark. If that one is unblocked, the run
will most likely *finish* and report success, which is worse:
- RLCD trains with a KL anchor that is numerically zero (look for `loss_kl ≈ 1e-6` in
  `wave7_rlcd_seed*/train_log.jsonl`).
- `verify_pull.py` will call the pull INCOMPLETE.
- Any resubmit after RLCD's first save will either be refused behind a masked exit 0, or silently reset
  the RLCD adapter to the base model and save checkpoints that `decision_eval` loads as the base.

## Minimal ordered fix list

1. **P0-1.** Owner decision on the mass invariant for trained adapters. In the meantime, log
   `min_label_mass` per SFT step, and for this job collect with `require_mass=0.0` plus a per-row `mass`
   field.
2. **P1-2.** `if [ "$RC" -ne 0 ]; then exit "$RC"; fi` after the RLCD wait loop.
3. **P1-1.** `main()` returns 75 unless `state.step ≥ steps−1`.
4. **P1-3.** `_Policy`: load **either** the resume checkpoint **or** the base adapter, never both.
5. **P1-4.** Precompute π_sft once at RLCD step 0 (eval, no_grad), persist it to `pi_sft.json` keyed by
   `labels_sha`, and reuse it on resume.
6. **P1-5.** Skip collect and relabel when `${POLICY_LABELS}.manifest.json` exists.
7. **P2-1.** `verify_pull.py` wave7 SFT expectations 159 → 120.
8. **P2-2.** `evaluate()` under `torch.no_grad()`, chunked by batch.
9. **P2-3 / P2-4.** Full-file label sha in the sbatch; refuse to resume an `aborted` state.
10. **P3.** `SFT_FINAL` from `state.json`; `|| true` on the `REPO_DIRTY` pipeline; python/CUDA sanity
    line; hoist `set(val_idx)`.

Before resubmitting, set `EXPECTED_COMMIT` (exported) to the commit carrying these fixes.
`8e53cbd` moved HEAD during this audit.
