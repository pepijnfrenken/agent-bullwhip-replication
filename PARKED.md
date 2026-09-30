# PARKED — waiting on TU/e compute response (as of 2026-09-21)

> **UNPARKED 2026-09-30 — compute landed: TU/e Umbrella (free institutional Slurm).**
> The three fix steps this document listed are all done, in order:
> **(1) compile path disabled** — `grpo_train.harden_compile_path()` sets `TORCHDYNAMO_DISABLE=1`
> before `torch` is imported and clears the transformers generation-compile knob after load, and
> the resolved knobs are printed into every run's log so the pilot log *proves* the state it ran in;
> **(2) rollouts batched** — `--rollout-batch` drives the 4 same-week roles × G episodes through one
> `generate()` call: at G=8/horizon=36 that is 72 calls per step at the default `--rollout-batch 16`
> (36 with the whole week in one call) instead of 1 152 single-prompt calls (the ~10× lever, now
> measured rather than assumed: `bench_rollout.py` reports µs/`generate()` and the
> batched-vs-sequential speedup in the pilot);
> **(3) hardware** — training moves to `tue.gpu3.q` (8× L4, 24 GB, ~300 GB/s), still €0.
> Plus walltime safety, which the desktop never needed: atomic checkpoints every N steps, `--resume`
> from `state.json`, and a SIGTERM handler so Slurm's pre-walltime signal checkpoints instead of
> losing an 18 h job.
> Run order and failure playbook: `hpc-umbrella/OPERATOR.md` (kit vendored in this repo since 2026-09-30).
> The parked Bug is still *hypothesis-first*: the pilot's step A is the first real execution of the
> torch path anywhere, on a matched-era module stack (torch 2.1.2) — if it runs, the version-pairing
> theory firms up; if it stalls with `TORCHDYNAMO_DISABLE=1` set, that is itself the result and the
> log + `py-spy` dump are the evidence to bring back. The rest of this document is the history of
> the park and is unchanged.

**Status:** HOLD as of 2026-09-21 — **superseded 2026-09-30, see the UNPARKED note at the top.**
Nothing is running anywhere. Desktop WSL is stopped; no processes; no money burning.

## Why we stopped
The Wave-6 smoke never completed a single training step on the desktop, and the attempt
ended in a host-disk incident (see below). Rather than fight an 8GB consumer GPU with a
software bug in it, the access question comes first.

## The blocker is NOT the model or the GPU — it's software

**Evidence (py-spy dump, `/home/pino/diag2.log`):** main thread parked in
`torch/_tensor.py:581 backward`, with many worker threads in
`torch/_inductor/compile_worker/subproc_pool.py:47 _recv_msg` / `:153 _read_thread`.
Corroborating: 21 `compile_worker` processes, GPU at 100% util but only ~80W, VRAM
oscillating 828 MiB ↔ 7886 MiB, `/tmp/torchinductor_*` holding 0 cached `.py` files, and
**no explicit `torch.compile` anywhere in `agent_bullwhip/`**.

**Hypothesis (unverified):** something (transformers 5.17.0 on torch 2.5.1+cu121 — a version
mismatch; transformers 5.x targets much newer torch) compiles per unique input shape, so
every new sequence length triggers a fresh compile and the process blocks on the compiler.

**Fix sequence when we resume (all free, local):**
0. Run the work through the mission runtime — `mr init/preflight/...` (see `~/projects/mission-runtime`),
   so this exact loss cannot happen again: event log + state + resume, artifacts verified against disk.
1. Disable the compile path — `TORCHDYNAMO_DISABLE=1` and/or the transformers generation-compile knob
2. Batch the rollouts — 32 sequential `generate()` calls → one batched call (the real 10x lever)
3. Only then think about hardware

## The host-disk incident (read before running anything long on the desktop)

The run died because **Windows C: hit ~0 bytes free**: WSL's root filesystem remounted
**read-only**, every external binary returned `Input/output error` (bash builtins still
worked), and then the distro refused to start at all (`Wsl/Service/CreateInstance/E_FAIL`).

**Root cause:** the Ubuntu vhdx is ~287.7GB and non-sparse on a 930GB C: with only ~7GB free,
so guest writes couldn't be materialised. Nothing looked wrong *inside* WSL — the guest
reported 670GB free the whole time. Classic WSL trap: the host runs out before the guest does.

**Recovery that worked, in order:**
1. Free host space — I cleared 13.1GB of pure garbage (`%LOCALAPPDATA%\Temp\wsl-crashes`
   held **10.7GB of WSL crash dumps**, plus temp junk). C: 7.15GB → 20.2GB free.
2. `wsl --shutdown`, then restart the distro → exec and writes healthy again.

**Preflight for any future run:** `df -h /mnt/c` — abort if under ~12GB free.
**Still available to reclaim (needs Pino's approval):** 76GB HuggingFace cache in the Windows
`.cache`, 5.2GB Recycle Bin, 4.6GB pip cache, 2.0GB Docker. Guest side: `abliteration-local`
= 193GB, `if-ablation` = 42GB, guest HF cache = 41GB.
**Only one physical disk** (Samsung 980 1TB) — moving the distro elsewhere isn't an option.

## Artifacts worth keeping (all intact)
| File | What it is |
|---|---|
| `/home/pino/wave6_smoke_report.md` | 112 lines — verified: PARAMS 596,049,920, bf16, 1.19GB VRAM after load; the timing split was never captured |
| `/home/pino/diag2.log` | py-spy dump of the stuck process — the core evidence |
| `/home/pino/diag_grpo.log`, `wave6_spy.log`, `wave6_repro.log` | earlier diagnostic runs |
| `/home/pino/tue-hpc-email.md` | the request email (Pino sends it himself) |
| `/home/pino/tue-supercomputing-form-answers.md` | same content mapped to the Topdesk form fields |
| `/home/pino/tue-snellius-request.md` | Snellius-specific draft + cluster caveats |

## Next direction once compute lands: a Jev-style calibrated decision model

Not a new architecture — a small delta on the existing loop:
1. Typed output contract `{decision, probability}` instead of free text
2. Calibration objective: reward = payoff − λ·(Brier/log-loss)
3. Confidence bands (act / flag / escalate) with thresholds in our code
4. A calibration eval harness: reliability curves, ECE, Brier decomposition, selective-risk/AURC

**Order of work:** build the **eval harness first**, and start with the cheapest possible
experiment — measure the *current* policy's calibration (ECE + reliability diagram on held-out
episodes) with no training change. That baseline is the motivation and the before/after story.

**Why the bullwhip test beats a public trading arena for this:** we control the data-generating
process (Poisson demand), so true probabilities are known and calibration can be scored against
ground truth. Financial generalization transfers the same machinery (orders → trades → risk flags),
with the honest caveat that non-stationarity means in-sample calibration will not transfer.
