# STAGING-READY — operator handoff: WAVE6-FIX + WAVE8 on the Umbrella cluster

**Written 2026-10-07** (mission `bullwhip-impl`). Both specs are implemented, tested and staged.
**Nothing has been submitted and nothing has been trained** — every number below is a hub-CPU
pre-flight read or a code fact. This doc is the checklist: what to pull, what to run, in what
order, how long it takes, what the gates read, and what comes back.

Repo: `~/projects/agent-bullwhip-replication` (hub) → `$BULLWHIP_ROOT/agent-bullwhip-replication`
(cluster, `BULLWHIP_ROOT=/scratch-shared/$TUE_USER/bullwhip`, `KIT_DIR=$BULLWHIP_ROOT/kit`).
Git is the **only** transfer channel in both directions. Suites at this revision: `uv run python -m
pytest -q` → **481 passed, 20 skipped** (hub venv, 47.9 s; measured 2026-10-07 during this write-up).

The specs stay the pre-registration: `waves/WAVE6-FIX_SPEC.md`, `waves/WAVE8_SPEC.md` (their Changes
logs carry every implementation deviation). Per-wave operator blocks are already in
`../hpc-umbrella/OPERATOR.md`; this doc is the ordered, go/no-go version of those.

---

## 0. Read this first

### 0.1 What is already verified vs what only the cluster can produce

| item | where | status |
|---|---|---|
| F0 grid (revised `w8_g26b`) | hub, `results/wave8/f0.json` | **PASS** (fresh seeds; numbers §1) |
| F0 grid (pre-registered `w8_g26`) | hub, `results/wave8/f0_w8_g26_prereg.json` | **FAIL** — kept as the record; superseded before any training |
| F1 teacher check | hub, `results/wave8/f1.json` | **PASS** (PI1 4,591 vs floor 8,541, 30/30) |
| `S_nom` sample + oracle labels | hub, `results/wave8/S_nom{,_oracle}.json*` | committed (sha256 `f651fc58…`, oracle modal letter A, m\* 0.1270) |
| **F2 interface** (26 single-token labels, ids 32..57, prompt SHA) | needs the real tokenizer snapshot | **only via the cluster pre-flight job** (§2.3) |
| **F3 control stats** (untrained-readout entropy quantiles, I0) | needs the model on a GPU | **only via the cluster pre-flight job** (§2.3) |
| WAVE6-FIX V0–V3 + R-rules | needs 6 GPU runs | not run |
| WAVE8 G1–G7 | needs R1 (and maybe R2) runs | not run |

`gpu-wave8-train.sbatch` **refuses to start** unless `results/wave8/gates.json` has F0 and F1 passing
*and* the rung's gates file has F3 frozen. So §2.3 is not optional and cannot be skipped by accident.

### 0.2 Order of operations

```
login node:  pull → stage.sh → config.sh --check
             ├── WAVE6-FIX : sbatch --array=0-5 gpu-w6fix.sbatch          (§2.2)
             └── WAVE8     : sbatch  gpu-wave8-preflight.sbatch            (§2.3, F2+F3)
                             sbatch --array=0-5 gpu-wave8-train.sbatch      (§2.4, R1)
                             sbatch gpu-wave8-eval.sbatch                   (§2.5)
                             ├─ only if R1 fails G1: R2 path                (§2.6)
                             └─ only if budgets/EVAL allow: ladder          (§2.7)
hub, after results: GGUF/CPU step                                        (§2.8)
```

WAVE6-FIX and WAVE8 are independent; either may be run first. Do **not** run R2 and R1 at the same
time on the 0.6B lane: R2's decision depends on R1's G1 read (§2.6).

### 0.3 What comes back through git, and what stays text-only

**Through `push_results.sh`** (cluster-side, forces the `git add -f` that `results/` needs) — see §4
for the exact per-step path lists.
**Text-only (not pushed by the script):** the Slurm stdout/stderr under `$BULLWHIP_ROOT/logs/` —
copy the ones that matter into `results/wave8_logs/` or `results/w6fix_logs/` **first**, then pass
those paths to `push_results.sh` (that convention is what puts them in the repo).
**Stays on the cluster (regenerable, untracked):** `checkpoint-*/` adapter dirs (`.gitignore`),
`round<r>/` working state/label files, `hf/`, `venv/`, `*.gguf` + the merged HF dir.

---

## 1. Pre-flight results already recorded (hub, committed)

### 1.1 F0 — choice-set grid gate (`results/wave8/gates.json` → `F0`, `F0_prereg_w8_g26`)

Thresholds (WAVE8 §6.1, unchanged): relative J-regret vs the 0..200/2 reference **≤ 0.5 % mean** and
**≤ 2 % p90** in both regimes; top-letter (Z) target mass **≤ 0.06** mixed / **≤ 0.03** nominal;
target-argmax at Z **≤ 0.08** mixed / **≤ 0.03** nominal.

| read | grid | mixed regret mean / p90 | mixed Z-mass / Z-argmax | nominal Z-mass / Z-argmax | verdict |
|---|---|---|---|---|---|
| 1,000 states, state seed 1, label seed 6000 | `w8_g26` (pre-registered) | 0.236 % / 0.569 % ✓ | 0.0684 ✗ / 0.098 ✗ | (nominal) / 0.032 ✗ | **FAIL** |
| 1,000 states, state seed 3, label seed 6050 | `w8_g26b` (staged) | 0.109 % / 0.340 % ✓ | 0.0404 ✓ / 0.058 ✓ | 0.0154 ✓ / 0.021 ✓ | **PASS** |

The revision is the one §3.1 of the spec pre-authorised ("if F0 fails at ≥ 1,000 states the grid is
revised before any training"). No threshold was moved. **Caveat carried into the results doc:** the
reference grid ends at 200, so for a grid whose cap *is* 200 the regret column cannot see truncation
beyond 200; that dimension is carried by the Z-mass / Z-argmax gates.

`w8_g26b` (staged, `results/wave8/grid_w8_g26b.json`; generated by `scripts/wave8_grid.py`):

```
A..P  =  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30        (unchanged from wave 7)
Q..Z  = 36 44 54 64 78 94 114 136 166 200                       (generated geometric tail)
token ids A..Z = 32..57 (contiguous, single-token); prompt SHA 55bdc663…  (both renderers)
```

A..P keeping their wave-7 values matters: the untrained readout's letter-E = order-8 prior (the strong
baseline) stays the *same decision*, so C1-style comparisons stay interpretable.

### 1.2 F1 — teacher check (`results/wave8/f1.json`)

Closed-loop PI1 policy (each role plays `argmin_a E[J]` on `w8_g26b`, M=16) vs the continuation floor
(θ=3.5, λ=0.2) on 30 curriculum paths, seeds **9000..9029** (never eval paths):

* PI1 mean cost **4,590.7** vs floor mean **8,541.2**; PI1 cheaper on **30/30** paths → **PASS**.
* Basis: the label `J(s,a)` is "one-step improvement over a slow floor", so if PI1 could not beat its
  own continuation there, no learner could be read as capacity-limited (WAVE8 §2.1). It can.

### 1.3 `S_nom` + oracle (`results/wave8/S_nom.jsonl`, `S_nom_oracle.json`)

Fixed 1,024-state nominal-floor sample (16 episodes, seeds 9100..9115, weeks 0..33, all 4 roles,
stride 2), sha256 `f651fc58e5a34c36421848a09e91090275a3e4786743549afb4ad45f0bc6aaa8`. Oracle labels
(M=16 on `w8_g26b`): the most common argmin letter is **A (order 0)**, share `m* = 0.1270`, Wilson
upper bound `m_hi = 0.1487`. These two numbers are the collapse gate's `m*`/`m_hi` inputs; the
entropy quantiles `H_lo/H_hi` and `I_hi0` are frozen by the cluster pre-flight (§2.3) on the
*untrained* readout, before any training — that ordering is the point of F3.

Everything in §1 arrives on the cluster with the first `git pull`. Nothing else is needed to start.

---

## 2. Step-by-step (login node), in order

All sbatch files are submitted **from the kit dir** (`SLURM_SUBMIT_DIR` must be `${KIT_DIR}`), and
`EXPECTED_COMMIT` must be **exported** (Slurm's `--export=ALL` forwards only exported variables).

### 2.1 Pull + stage

Hardcode the commit you intend to run **from the hub**, before staging (do not compute it on the
cluster *after* the pull — that is circular):

```bash
# on the HUB, after the final push:
git -C ~/projects/agent-bullwhip-replication rev-parse HEAD     # -> $EXPECTED_COMMIT
```

Then on the login node:

```bash
B=/scratch-shared/$USER/bullwhip; R=$B/agent-bullwhip-replication
export EXPECTED_COMMIT=<the sha printed on the hub>

# 1. put the repo on that commit (first time: git clone; then:)
git -C "$R" fetch --all --prune && git -C "$R" checkout master && git -C "$R" pull --ff-only
test "$(git -C "$R" rev-parse HEAD)" = "$EXPECTED_COMMIT" || echo "STOP: cluster HEAD != $EXPECTED_COMMIT"

# 2. stage from the freshly pulled repo copy (mirrors hpc-umbrella/ -> $B/kit, re-execs there,
#    venv + weights idempotent). Run it from $R/hpc-umbrella, NOT from $B/kit:
#    stage.sh mirrors its own directory into kit/ BEFORE it pulls.
bash "$R/hpc-umbrella/stage.sh"
bash "$B/kit/config.sh" --check          # must end "RESULT: 0 fail / N warn"

# 3. prove the kit is the repo (not a stale mirror), then submit from the kit
diff -q "$B/kit/gpu-wave8-train.sbatch" "$R/hpc-umbrella/gpu-wave8-train.sbatch" && echo "kit == repo"
cd "$B/kit"
```

Stage also runs a torch-free pytest pass and one GPU-free `import torch,transformers,peft` check —
the first place the **gated torch tests** (`tests/test_w8_train_torch.py`) actually execute. If they
fail there, stop and report: they cover the adapter/anchor paths the hub cannot run.

### 2.2 WAVE6-FIX (6 array tasks, `gpu-w6fix.sbatch`)

Arms = rows of `campaign-w6fix.tsv`: `a0, a0b, a1, a1_seed2, a2, a2_seed2`. `tue.gpu.q -C l4`, 1 GPU,
8 CPU, 16 G, **walltime 14 h**, `--signal=B:TERM@600`, `--resume` on resubmit.

```bash
cd "$B/kit"
# optional pilot: 4 real steps of A1 (checks contract parsing + the step-0 dump shape)
W6FIX_STEPS=4 sbatch --array=2 gpu-w6fix.sbatch
# then the probe
sbatch --array=0-5 gpu-w6fix.sbatch
squeue --me
```

Early stop is pre-registered: an A1/A2 run whose P50 `contract_parse_rate` is < 0.02 at step 50 stops
itself (a low rate there cannot reach 0.20 by step 100). Those arms then report INCOMPLETE in
`verify_pull.py` **by design**. A `n_nonfinite > 0` abort is a run failure: rerun that one from
scratch with a new `--out` (never `--resume`).

### 2.3 WAVE8 pre-flight — F2 + F3 (MANDATORY before any training)

One short GPU job (`tue.gpu.q -C l4`, 1 GPU, 8 CPU, 16 G, **walltime 1 h**) that (a) checks the real
tokenizer snapshot for F2 and (b) freezes the untrained-readout control stats on `S_nom` into
`results/wave8/gates.json` as F3. It refuses to overwrite an existing F3 (use `W8_FORCE_F3=1` only if
you mean to re-freeze).

```bash
cd "$B/kit"
sbatch gpu-wave8-preflight.sbatch           # rung 0.6b (default)
python3 "$R/scripts/wave8_gates.py" report --gates results/wave8/gates.json
# expect: F0 pass, F1 pass, F2 prompt_sha_equal true, F3 present
```

F2 for `w8_g26b` should print `prompt_sha_equal: true` against `55bdc663…` (the hub value; if the
cluster venv's template bytes differ, the job says so loudly — that is the gate, not a nuisance).

### 2.4 WAVE8 R1 — outcome-scored RL (6 array tasks)

`--array=0-5` enumerates `(lr, seed)`: **lrs {1e-5, 5e-5} × seeds {0,1,2}**. Each task is one resumable
job: 3 rounds of collect → relabel → train (170 steps) → collapse read, then the curriculum-validation
eval (band 9500..9529, used to pick the lr — never eval paths). `tue.gpu.q -C l4`, 1 GPU, 8 CPU, 16 G,
**walltime 8 h**.

```bash
cd "$B/kit"
sbatch --array=0-5 gpu-wave8-train.sbatch
squeue --me
```

Watch list per task log (`logs/wave8-train_<A>_<a>.log`):

* `repo … HEAD=$EXPECTED_COMMIT dirty_files=0`, then `pre-flight ok: … F3 frozen in …`.
* `=== wave8 r1 task <i>: lr=… seed=… rounds=3x170 ===` appears immediately; each round prints its
  collect / relabel / train stages and a collapse read.
* `results/wave8_r1_lr<l>_seed<s>_round<r>/train_log.jsonl`: 170 steps each; `std_j_clamped_frac`
  is logged per RLCD step (how flat `J` was on that round's states).
* `results/wave8_r1_…/STOPPED_collapse.json` = the two-consecutive-collapse stop-loss fired — that is
  a **result**, not a failure (exit 0 by design). Report it as "stopped: collapse".
* Exit **75** = Slurm SIGTERM (partial) → **resubmit the same array task**; `round_state.json` +
  `--resume` continue at the interrupted stage.

### 2.5 WAVE8 eval (1 job)

Per rung: the untrained control (C0) **and** each trained adapter as step-argmax (n=1),
step-sampled (n=30) and noisy-argmax (n=30) — all with per-decision `traces` filled — plus the
permutation probe and the collapse-final read with the calibration table. `tue.gpu.q -C l4`, 1 GPU,
8 CPU, 16 G, **walltime 4 h**.

```bash
cd "$B/kit"
sbatch gpu-wave8-eval.sbatch        # default W8_EVAL_ARMS="r1"; add r2sft/r2rl only after §2.6
# W8_EVAL_ARMS="r1 r2sft r2rl" W8_R2_LR=<winner> sbatch gpu-wave8-eval.sbatch
```

The job selects the R1 lr itself from `results/wave8_val/` (`scripts/wave8_analysis.py select-lr`),
unless you override with `W8_LR=`. Gate readout afterwards, on the hub (§4).

**If the eval exits 1 within seconds, it is a refusal, not a crash.** The three refusals and their fixes:
`FATAL: … missing (F3 not frozen for this rung)` → run `gpu-wave8-preflight.sbatch` for that `W8_RUNG`;
`FATAL: missing train output results/wave8_r1<SUF>_<lr>_seed<S>_round3/state.json` → the eval guessed the lr
from `results/wave8_val/` and it does not match the tag the train job wrote (or that seed never reached round 3):
resubmit with the explicit pair, e.g. `W8_LR=5e-5 W8_EVAL_SEEDS="0 1" sbatch gpu-wave8-eval.sbatch`;
`FATAL: EXPECTED_COMMIT` → stale checkout on the cluster.
One-command evidence bundle: `bash hpc-umbrella/diagnose-wave8.sh eval` (`ssh <user>@hpc.tue.nl 'bash -s' <
hpc-umbrella/diagnose-wave8.sh eval > wave8-diag.txt`) — exit codes, the error lines, and the dir/adapter inventory
that shows the mismatch directly.

### 2.6 R2 — only if R1 fails G1 (decision table §6.6)

R2 = soft-target SFT warm start (one full epoch) + the same RL rounds anchored to the frozen SFT
adapter. **Skip it entirely if R1 passes G1 on ≥ 2/3 seeds.** Two extra steps, in this order:

```bash
cd "$B/kit"
sbatch cpu-wave8-label.sbatch                          # 26-candidate round-1 labels (tue.cpu3.q, 96 cores)
W8_ARM=r2sweep sbatch --array=0-1 gpu-wave8-train.sbatch   # 500-step lr sweep {1e-5, 1e-4}, val every 100
python3 "$R/scripts/wave8_analysis.py" select-sft-lr       # -> winner on validation NLL
W8_ARM=r2 W8_R2_LR=<winner> sbatch --array=0-2 gpu-wave8-train.sbatch   # SFT + sft-gates + RL rounds
```

The label job is **not** needed for R1 (R1 labels its own on-policy states each round).

### 2.7 Ladder — 1.7B then 4B (after the 0.6B R1 completes without an infrastructure fault)

Same R1 recipe at the next rung, with the 0.6B **validation-selected** lr:

```bash
bash "$B/kit/stage.sh" --with-ladder                    # login node; downloads 1.7B + 4B (4B ~8 GB)
cd "$B/kit"
W8_RUNG=1.7b sbatch gpu-wave8-preflight.sbatch          # per-rung F2/F3 (the untrained readout differs)
W8_RUNG=1.7b W8_LRS=<winner> sbatch --array=0-2 gpu-wave8-train.sbatch
# 4B: micro-batch smoke FIRST (fit on a 22 GiB L4 is unverified)
W8_RUNG=4b W8_MICRO_BATCH=8 W8_LRS=<winner> W8_SEEDS=2 sbatch --array=0-1 gpu-wave8-train.sbatch
W8_RUNG=1.7b sbatch gpu-wave8-eval.sbatch               # W8_EVAL_SEEDS="0 1 2"; 4B: "0 1"
```

Rung revisions are pinned in `config.sh` (`MODEL_REVISION_17B=70d244cc…`, `MODEL_REVISION_4B=1cfa9a72…`,
read from the HF API 2026-10-07). If a rung cannot fit or fails on the L4, report it
"not run: \<reason\>" — the spec forbids substituting another model.

### 2.8 GGUF / CPU step (hub, after a trained adapter exists)

Not a cluster step. On the hub, with an adapter from §2.5 (and llama.cpp already built at
`/home/pino/bw6-eval/llama.cpp`):

```bash
cd ~/projects/agent-bullwhip-replication
python3 scripts/wave8_gguf_serve.py convert --adapter <round3 ckpt dir> --out results/wave8_gguf/<arm>
# then start llama-server on the f16/Q8/Q4 artifact and, for each quant:
python3 scripts/wave8_gguf_serve.py torch-probs --states results/wave8/S_nom.jsonl --adapter <ckpt> --out results/wave8_gguf/torch.jsonl
python3 scripts/wave8_gguf_serve.py probs       --states results/wave8/S_nom.jsonl --url http://127.0.0.1:8080 --out results/wave8_gguf/f16.jsonl
python3 scripts/wave8_gguf_serve.py equivalence --torch results/wave8_gguf/torch.jsonl --gguf results/wave8_gguf/f16.jsonl --quant f16 --out results/wave8_gguf/equiv_f16.json
python3 scripts/wave8_gguf_serve.py eval        --url http://127.0.0.1:8080 --outdir results/wave8_gguf --tag-prefix w8-gguf
```

**Pre-registered equivalence gate:** argmax agreement with the torch readout ≥ 0.99 (f16, top-1 prob
within 0.02), ≥ 0.98 (Q8_0), Q4_K_M reported; `|mass_gguf − mass_torch| < 0.01`. No CPU latency or
cost number may be quoted before this job measures it. `*.gguf` and the merged HF dir are untracked;
`manifest.json` (with sha256) plus the summaries are the evidence.

---

## 3. Gate tables (the go/no-go reads)

### 3.1 WAVE6-FIX (after all six tasks; both seeds must agree in direction, else "mixed")

Validity first — a probe that fails a V-gate is **void**, not negative:

| gate | condition |
|---|---|
| V0 reproduction | A0 re-run mean parse-fail steps 0–49 ∈ [0.81, 0.87] (historical 0.8401 ± 0.013) |
| V1 prompt bytes | A0 prompt bytes == pre-change golden SHA; A1/A2 == `apply_chat_template(…, enable_thinking=False)` |
| V2 scoring | A1/A2 `clip_frac`, `max_abs_logratio` at A0's bf16 noise floor (≈0.003) |
| V3 parser | `parse_contract` golden-string unit tests pass (`tests/test_w6fix_contract.py`) |

Then (P50 = window mean `contract_parse_rate`):

| rule | condition | go/no-go |
|---|---|---|
| **R-A** contract was the blocker | A1 P50 ≥ 0.90 (both seeds) | full-length fidelity run (A1 flags, 300 steps × 2 seeds); wave-6 verdict reclassified "interface artifact" |
| **R-B** reward also needed | A1 P50 < 0.90 **and** A2 P50 ≥ 0.90 and A2−A1 ≥ 0.20 | full-length run uses A2's reward; deviation (iv) declared downstream |
| **R-C** 0.6B cannot hold the contract | A1 P100 < 0.20 and A2 P100 < 0.20 | reassign the fidelity claim to the size ladder (1.7B, then 4B); wave-6 stays "non-informative" |
| **R-D** middle | anything else | one knob at a time, 50 steps, seed 12345: K1 → K2 → K3 → K4 → K5 |

Step-0 read (H1): if A1's first-5-step `contract_parse_rate` is already ≥ 0.90, format compliance is
untrained and R-A holds immediately (the arm still runs to step 100 for the cost trajectory).

### 3.2 WAVE8

Pre-flight (all must pass; the train sbatch enforces them):

| gate | quantity | threshold | status |
|---|---|---|---|
| F0 grid | regret / Z-mass / Z-argmax | §1.1 | **PASS** (revised grid) |
| F1 teacher | PI1 vs floor, 30 paths | PI1 ≤ floor | **PASS** (4,591 vs 8,541, 30/30) |
| F2 interface | 26 single-token labels, ids 32..57, prompt SHA | asserts | cluster job §2.3 |
| F3 control | `H_lo,H_hi,m_hi,I_hi0` frozen pre-training | recorded in `gates.json` | cluster job §2.3 |

Computed gates per arm (all in `scripts/wave8_analysis.py` → `results/wave8_report.{md,json}`):

| gate | read | verdict rule |
|---|---|---|
| collapse | median entropy vs `H_lo`/`H_hi`, modal-letter share vs `m_hi`, state-dependence `I` vs `I_hi0` | collapsed iff `[median H < H_lo and (modal share > m_hi or I ≤ I_hi0)]` or `[median H > H_hi and I ≤ I_hi0]`; a collapsed arm's costs are reported but cannot satisfy SUPPORTING |
| permutation | argmax value unchanged under 3 shuffled grids, 512 `S_nom` states | ≥ 0.90 passes; a fail withholds "reads the options" and labels the arm "fixed-ordinal-scale specialist" (cost gates still decide) |
| **G1** (binding) | vs own untrained readout, paired **noisy-argmax n=30** | median cost ↓ ≥ 15 % **both** ways **and** ≥ 20/30 wins **and** one-sided sign-test p < 0.02; a seed passes iff all three; arm needs ≥ 2/3 seeds |
| G2 | vs `wf_tuned` noisy floor 5,346 | tiers: strong ≤ 4× (21,384), headline ≤ 2× (10,692), floor-parity ≤ 1.25× (6,683). Untrained readout is 11,569 = 2.16× → **G1, not G2, is the test of learning** |
| G3 | vs wave-6 `a0_base` on identical paths | ≥ 20/30 wins; a pass is "no rogue tail", never "cheaper policy" |
| G4 | paper-direction triplet on step-sampled n=30 | mean ↓ ≥ 20 %, CV ↓ ≥ 25 %, max ↓ (untrained sampled readout: median 22,207, CV 0.48) |
| G5 | reproducibility | seed-pair mean gap within 2·√(SE_i²+SE_j²); zero-variance arms flagged "mechanically disagreeing" |
| G6 | collapse + permutation | read for every arm |
| G7 | parse/mass | `parse_fail == 0` (asserted); `min_mass` recorded, ≥ 0.9 expected for adapters, < 0.9 flagged |

Decision table (WAVE8 §6.6):

| outcome | reading | next |
|---|---|---|
| F0–F3 pass, R1 passes G1 (≥ 2/3), collapse clean | SUPPORTING candidate | check G4 + permutation; run ladder; GGUF; publish "specialist decision model works at 0.6B" with the §10 bounds |
| R1 fails G1, collapse tripped with **edge** signature | grid/teacher story still live | run R3 once (R3b/R3c variants are in the grid registry; R3a is **not** implemented — it needs per-state grids through the whole pipeline) |
| R1 fails G1, collapse tripped **mid-letter** | letter-lock = objective/lr, not grid | run R2 (warm start); if R2+RL also locks → INCONCLUSIVE-negative at 0.6B, go to the ladder |
| R1 fails, collapse clean, perm fail | value-reading not learned in budget | R2 with `--permute-p 1.0` on seed 0 only; otherwise report |
| F1 fails | teacher defective | stop; revise the labeler continuation (not a model result) |
| fails at every rung, pre-flight clean | CONTRADICTING (bounded) | write the honest negative; STATE.md update |

Stop-loss already wired in: two consecutive collapse trips at checkpoint reads stop that run (marker
`STOPPED_collapse.json`); all three seeds of the better lr tripping by end of round 2 stops R1 as a
whole; the whole wave is capped at **≤ 70 GPU-h**.

---

## 4. Send up + verify (git, both directions)

On the cluster, after the jobs finish — copy the Slurm logs into the repo first, then push:

```bash
cd "$R"
mkdir -p results/wave8_logs results/w6fix_logs
cp "$B"/logs/wave8-*.log results/wave8_logs/
cp "$B"/logs/w6fix_*.log results/w6fix_logs/

bash hpc-umbrella/push_results.sh -m "w6fix runs (job <id>)" \
     results/grpo_w6fix_* results/w6fix_logs
bash hpc-umbrella/push_results.sh -m "wave-8 R1 + eval (job <id>)" \
     results/wave8_r1_* results/wave8_val results/eval results/wave8 results/wave8_logs
# R2 only:  also results/wave8_r2sft_* results/wave8_r2rl_* results/wave8_labels
# ladder:   also results/wave8_r1-1.7b_* results/wave8_r1-4b_* results/wave8/gates_1.7b.json results/wave8/gates_4b.json
```

On the hub:

```bash
cd ~/projects/agent-bullwhip-replication
git pull --ff-only
python3 hpc-umbrella/verify_pull.py ~/projects/agent-bullwhip-replication --size
#   EXPECT covers the 18 R1 round dirs (170 steps each) and the 6 w6fix arms; R2/R3/ladder dirs
#   are reported as "extras (fine)". Early-stopped w6fix arms report INCOMPLETE by design.
python3 scripts/w6fix_analysis.py --out results/w6fix_report.md --json results/w6fix_report.json
python3 scripts/wave8_analysis.py --out results/wave8_report.md --json results/wave8_report.json
```

`verify_pull.py` is **lax, not wrong**: it checks step counts, not gate outcomes. The gate outcomes
are §3.

---

## 5. Expected wall-clock (measured bases; `[INFERENCE]` where marked)

| step | partition | walltime limit | expected |
|---|---|---|---|
| WAVE6-FIX array, 6 tasks in parallel | `tue.gpu.q -C l4` | 14 h each | 110–190 s/step `[INFERENCE]` → ≈ 3–6 h wall total, ≈ 14–24 GPU-h |
| WAVE8 pre-flight (F2+F3) | `tue.gpu.q -C l4` | 1 h | minutes (tokenizer check + 1,024 states) |
| WAVE8 R1, 6 tasks in parallel | `tue.gpu.q -C l4` | 8 h each | **≈ 1.1 h per (lr, seed)** = 170×3 steps × ~4.7 s + 3 collects (~1.5 min) + 3 relabels (~7 min on 8 CPU) → ≈ 7 GPU-h total |
| WAVE8 labeler (R2 only) | `tue.cpu3.q`, 96 cores | 1 h | ≈ 4 min `[INFERENCE]` (wave-7 186.7 s at 21 candidates, scaled 26/21) |
| WAVE8 R2 (SFT + RL) | `tue.gpu.q -C l4` | 8 h | ≈ 3.7 h SFT (2,827 steps) + 3 RL rounds; lr sweep 1.3 GPU-h extra |
| WAVE8 eval | `tue.gpu.q -C l4` | 4 h | ≈ 50 min for 3 modes × (control + 3 seeds) + probes |
| Ladder 1.7B (3 seeds) | `tue.gpu.q -C l4` | 8 h | ≈ 1.8 h/seed `[INFERENCE]` → ≈ 5.5 GPU-h |
| Ladder 4B (2 seeds, micro-batched) | `tue.gpu.q -C l4` | 8 h | ≈ 4.4 h/seed `[INFERENCE]` → ≈ 9 GPU-h |
| GGUF + CPU serving | hub CPU | — | convert/quantize minutes; equivalence 2,000 states; full games |

If any task nears 10 min before its limit, `TERM@600` fires → exit 75 → resubmit that array task.

---

## 6. Failure / stop conditions

* **`FATAL: EXPECTED_COMMIT=… but HEAD=…`** → the checkout is not the staged commit. Fix the pull;
  do not unset the gate.
* **`FATAL: pre-flight ['F0'] not passing` / `lacks F2/F3`** → run §2.3 first (or the rung tag is
  mismatched: `W8_RUNG` selects both the model and the gates file).
* **exit 75** → SIGTERM-partial; resubmit the same array task (`--resume` continues).
* **exit 1** → a refusal or a Python exception. Refusals read clearly (labels sha changed, missing
  `pi_sft.json`, aborted non-finite run); read the traceback, do not blind-resubmit.
* **exit 3** → non-finite abort; the run refuses to resume. Start that one from scratch, new `--out`;
  a second abort ends that arm.
* **`STOPPED_collapse.json`** → the stop-loss fired by design, exit 0. Record as "stopped: collapse".
* **CUDA OOM on the L4 (22 GiB)** → the sbatch already passes `--grad-ckpt 1` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; for 4B set `W8_MICRO_BATCH`.
* **`dirty_files != 0` in a job log** → the cluster checkout was modified; that run is not
  reproducible against the pinned commit. Investigate before reading its gates.

---

## 7. Residual risks and what is unobserved here

1. **The torch `_Policy` paths are unexecuted on this hub** (no `peft` in any local venv): fresh-init
   LoRA, the base anchor (`model.disable_adapter()`), permuted RLCD steps, micro-batching. They are
   covered by gated tests (`tests/test_w8_train_torch.py`, `tests/test_wave7_adapters.py`) that run
   during `stage.sh` on the cluster venv. `[UNVERIFIED]` until that stage log exists.
2. **4B LoRA batch-32 fit on a 22 GiB L4** is arithmetic, not measured. Do the 10-step
   `W8_MICRO_BATCH` smoke before the full 4B run. `[INFERENCE]`
3. **The real GGUF conversion/serving path is unexercised** — `wave8_gguf_serve.py`'s serving and
   equivalence logic are tested against a fake llama-server only, and there is no trained adapter to
   convert yet. `[UNVERIFIED]`
4. **F0's reference cap is 200**, the same value as the staged grid's last letter. Truncation beyond
   200 is carried only by the Z-mass / Z-argmax gates; recorded in the spec Changes log.
5. **`w8_g26` (pre-registered) stays in the registry as a failing artifact.** The staged pipeline
   defaults to `w8_g26b`; do not pass `--grid w8_g26` anywhere.
6. **`pi_sft.json` mass erosion.** Wave-7 SFT eroded the untrained readout's label mass 0.997 →
   0.963–0.973. For adapters this is *recorded, not asserted* (`min_mass ≥ 0.9` is the flag); a value
   below 0.9 means the readout is leaving the label set — record it, it is a result.
7. **The `wait`-loop race from W7-READY Item R1** (an instantly-failing child can be masked as
   `RC=0`) is present in the wave-7 sbatch idiom. The wave-8 sbatch keeps the same shape; the
   window is exec/loader-level failures only. Not fixed; unchanged risk.
8. **Nothing here has been run.** Every "expected" number above is marked with its basis; no cost,
   parse rate, or gate outcome exists for either wave yet.
