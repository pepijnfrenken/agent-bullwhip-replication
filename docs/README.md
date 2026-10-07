# How to navigate this repo

Entry point for a fresh reader or a resuming mission. What this project is, where everything
lives, naming conventions, and how to run the pieces. Current standing of the work:
[`STATE.md`](STATE.md). Root [`../README.md`](../README.md) is the one-paragraph front door;
this file is the map.

---

## 1. What this project is

A replication + extension of **arXiv:2605.17036** ("Reliability and Effectiveness of Autonomous
AI Agents in Supply Chain Management") on the four-echelon MIT Beer Game. LLM agents play order
decisions; the paper claims GRPO RL post-training makes them reliable. This repo tests cheap
inference-time structure instead, then audits its own results. The project spans **7 waves** —
see the wave table in [`STATE.md`](STATE.md).

Research protocol (non-negotiable, from `waves/protocol.md`): **pre-register claims before running
them; every number in every doc must point at an on-disk artifact under `../results/`; audit before
publish; honest negatives are the deliverable.** Read root `../README.md` §research protocol before
writing any claim into this repo.

## 2. Top-level layout

```
agent-bullwhip-replication/
├── README.md              entry point (short statement + protocol + links)
├── CHANGES.md             running change log (chronological, newest first). Append, never rewrite.
├── waves/protocol.md            the research protocol
├── docs/
│   ├── README.md          ← you are here
│   ├── STATE.md           wave table, current verdicts, open issues
│   ├── waves/             wave docs + specs (WAVE*.md, waves/W7-READY.md, *_SPEC.md)
│   ├── audits/            audit docs (AUDIT*.md, WAVE7-AUDIT-*.md, AUDIT-RUNTIME-*)
│   ├── fixes/             infrastructure fixes (FIXES*.md)
│   ├── verify/            verification docs (VERIFY*.md)
│   ├── missions/          historical mission logs (PROGRESS*.md)
│   └── archive/           obsolete-but-kept (README_v1..v3, README_ITERATION_LOG, …)
├── agent_bullwhip/        the package: engine, agents, runners, wave-7 decision train/eval
│   ├── engine.py          deterministic 4-echelon beer game (leak-free after AUDIT2)
│   ├── agents.py          LLM agent wrappers + prompt builders + fallback policy
│   ├── interleaved_runner.py   the audit-surviving LLM run protocol (waves 1–3)
│   ├── runner.py          SEQUENTIAL runner — DO NOT USE for claims (killed by Audit #1)
│   ├── decision_eval.py / decision_readout.py / decision_labeler.py / decision_train.py
│   │                      the wave-7 decision-native lane (closed-choice readout, SFT+RLCD)
│   ├── grpo_env.py / grpo_train.py / grpo_trl.py   wave-6 GRPO training
│   ├── jev_agent.py       wave-5 Jev decision agent (typed questions, no generation)
│   └── knowledge/         KB playbooks (GENERAL_KB.md, PROMPT_KB.md)
├── tests/                 326 tests (must stay green — run via `python -m pytest`)
├── scripts/               LIVE tooling used by the current workflow (see §4)
│   └── archive/           LEGACY one-off scripts from past waves (kept, not deleted)
├── hpc-umbrella/          the cluster kit — do NOT restructure (operators copy it verbatim)
│   ├── OPERATOR.md        run order + failure playbook (START HERE for cluster work)
│   ├── config.sh          single pins file; reusable sbatch scripts
│   └── umbrella-results/  pulled cluster results (text evidence tracked, binaries untracked)
├── results/               370MB of evidence (gitignored except wave-7 readouts; see §5)
├── tools/                 small dev tools (label_fingerprint.py)
└── .gitignore             note: results/ ignored; POSTS*.md must never be committed
```

## 3. Naming conventions

- **Waves**: `waves/WAVE5.md` / `waves/WAVE6_SPEC.md` / `waves/WAVE7.md` — wave number = research iteration.
  Current = wave 7. Missing wave numbers were folded into the audit docs or the README history.
- **Specs**: `*_SPEC.md` = pre-registered design (gates, controls, null clause). A spec with a
  "changes log" section is a living document.
- **Audits**: `AUDIT*.md` = adversarial self-review; every audit retires something. Read the
  audits before trusting the headlines — the surviving claim is the one that survived.
- **Verifications**: `VERIFY*.md` = independent recomputation of a wave's numbers from raw
  artifacts (e.g. `verify/VERIFY-W7-EVAL.md`).
- **Run lanes**: `../results/<lane>/` where `<lane>` names the run family (`noisy_leakfree`,
  `chaotic_seeded`, `wild_seeded`, `jev_protocol`, `wave7_sft_seed0`, …).
- **Per-run artifacts**: `<config>.jsonl` = full per-decision traces; `<config>.summary.json` =
  per-run stats. Wave-7 eval files: `w7-<arm>-<mode>-<pattern>.jsonl` + `.summary.json`, and
  `w7-<arm>-perm.json` for permutation probes.
- **Dated docs**: `fixes/FIXES-2026-10-01.md`, `verify/VERIFY-2026-10-01.md` — dated by incident, kept in
  `fixes/` / `verify/`.

## 4. Where things live (code, docs, results)

| what | where |
|---|---|
| engine + agents + runners | `../agent_bullwhip/` |
| wave-7 decision-native code | `../agent_bullwhip/decision_*.py`, `grpo_*.py` |
| wave docs + specs | `waves/` (waves/WAVE5.md, waves/WAVE6_SPEC.md, waves/WAVE7.md, waves/WAVE7_SPEC.md, waves/WAVE6-FIX_SPEC.md, waves/WAVE8_SPEC.md, waves/W7-READY.md, **waves/STAGING-READY.md** = the cluster operator handoff for waves 6-fix + 8) |
| audits | `audits/` (audits/AUDIT.md…audits/AUDIT5.md, WAVE7-AUDIT-1/2.md, AUDIT-RUNTIME-*) |
| fixes + verifications | `fixes/`, `verify/` |
| historical mission logs | `missions/` (missions/PROGRESS.md, missions/PROGRESS_DESKTOP.md) |
| live scripts | `../scripts/` (../scripts/wave7_analysis.py, ../scripts/bench_rollout.py, ../scripts/walkforward_floor.py, …) |
| legacy one-off scripts | `../scripts/archive/` (final_*.py, ../scripts/archive/wave2.sh, probe_*.py, trace analyzers, …) |
| cluster kit | `../hpc-umbrella/` (OPERATOR.md, config.sh, *.sbatch, push_results.sh) |
| raw evidence | `../results/` (see §5) |

## 5. results/ — the evidence layer

`../results/` is **gitignored by default**; only the wave-7 readout set (eval JSONL, labels, logs,
`../results/wave7_floors.json`) is tracked, force-added via the push script. Waves 1–5 evidence
(`noisy_leakfree/`, `chaotic_seeded/`, `wild_seeded/`, `jev_protocol/`, …) lives on disk only.
**Never delete anything under `../results/`** — readouts, labels, logs, and checkpoints are the
evidence the protocol requires. Regenerable pieces (checkpoint dirs, `*.pt`) are untracked;
see `ARTIFACTS.md` for the manifest of untracked-but-kept artifacts.

The **supported evidence sets** (filenames match the README tables):

| set | what it holds |
|---|---|
| `../results/noisy_leakfree/` | wave-3 leak-free noisy arm (10×4 configs) |
| `../results/chaotic_seeded/` `../results/wild_seeded/` | seeded chaotic/wild arms (post-AUDIT3) |
| `../results/jev_protocol/` `../results/jev_controls_*/` `../results/jev_noisy20/` | wave-5 Jev arms + controls |
| `../results/local-eval/` | wave-6 a0_base eval (Qwen3-0.6B, step/noisy, trained/untrained) |
| `../results/eval/` | **wave-7 protocol evals** (21 runs + 7 permutation probes) |
| `../results/wave7_labels/` `../results/wave7_logs/` `../results/wave7_{sft,rlcd}_seed{0,1,2}/` | wave-7 training + eval artifacts |
| `../results/wave7_floors.json` `../results/wave7_report.md/.json` | wave-7 floors + generated readout |

## 6. HOW TO RUN

Prereqs: the project uses `uv` (a `.venv` also exists). Python ≥3.10; `requirements.txt` for
`pip`-style installs. Env vars: `FREEINFERENCE_API_KEY` for LLM arms (model defaults to
deepseek-v4-flash). CUDA/torch only needed for wave-6/7 training paths.

### Tests (the gate)

```bash
uv run python -m pytest -q        # 326 passed, 15 skipped (baseline 2026-10-07)
```

> **Note:** `uv run pytest` alone fails at collection (`agent_bullwhip` not importable — the
> pyproject is poetry-style). Always use `python -m pytest`.

### Eval / analysis / floors (no API needed)

```bash
uv run python scripts/walkforward_floor.py     # walk-forward + oracle floor columns
uv run python scripts/tuned_floor.py           # tuned-floor sweep
uv run python scripts/wave7_analysis.py        # wave-7 gate readout → results/wave7_report.md
```

### LLM arms (API calls)

```bash
uv run python -m agent_bullwhip.interleaved_runner \
  --configs kb_pointer_verbal,kb_system_gated,order_up_to,baseline \
  --runs 10 --horizon 36 --pattern noisy --seed 7 --outdir results/<lane>
```

The sequential runner (`python -m agent_bullwhip.runner`) is **not** for claims — it's the
protocol Audit #1 deleted. Smoke tests/archaeology only.

### Wave-7 decision-native lane

Everything wave-7 (label → SFT → RLCD → eval) runs on the cluster via `../hpc-umbrella/` —
see `../hpc-umbrella/OPERATOR.md`. Hub-side analysis: `../scripts/wave7_analysis.py`.

### Training (wave 6 GRPO)

Runs on TU/e Umbrella via the kit; `../hpc-umbrella/gpu-campaign.sbatch` drives the pre-registered
configs from `../hpc-umbrella/campaign.tsv`. Local dev: `../agent_bullwhip/grpo_train.py --help`.

## 7. Results flow hub ↔ cluster (the hard rule)

**Results travel through this git repo, in both directions. rsync is NOT used.** From `../hpc-umbrella/OPERATOR.md`:

- On the cluster: `bash hpc-umbrella/push_results.sh -m "<message>" <paths…>` —
  `git add -f` (results/ is gitignored) + commit + push.
- On the hub: `git pull --ff-only`.
- `../hpc-umbrella/pull_results.sh` exists only as an emergency bulk mirror of expiring scratch.

The commits record which job produced what — that provenance is why this rule exists.
`../hpc-umbrella/verify_pull.py` can verify pulled artifacts. Because `../hpc-umbrella/push_results.sh` force-adds,
the repo's tracked set includes some regenerable binaries; P4 of the 2026-10-07 cleanup
untracked the checkpoint dirs (see `ARTIFACTS.md`). New result pushes should pass text
readouts (JSONL/summaries/logs), **not** checkpoint dirs.

## 8. If you are resuming a mission

- `mr status` / `mr resume` per the mission runtime protocol (`../.omp/rules/mission-runtime.md`).
- Historical mission logs: `missions/` (wave-6 era; `missions/PROGRESS.md`, `missions/PROGRESS_DESKTOP.md`).
- The most recent full record of "where things stand": `STATE.md` + `../CHANGES.md` head.