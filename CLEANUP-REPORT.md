# CLEANUP-REPORT — repo reorganization + navigation (2026-10-07)

Mission: `bullwhip-cleanup` (evidence captures in `/home/pino/missions/bullwhip-cleanup/`).
Constraints honored: **no deletion** (66 checkpoint files + 1 adapter untracked, all still on
disk), **no git history rewrite**, **no numbers/verdicts/claims changed**, **no code-behavior
changes / no new deps / no restructuring of `agent_bullwhip/`, `tests/`, or anything inside
`hpc-umbrella/`** (only reference paths inside those trees were updated), `.env.local` untouched,
post drafts (`POSTS.md`, `POST_wave5_repost.md`) left untracked and uncommitted.

## 1. Before / after top-level tree

**Before (57 root entries):** 28 root `.md` docs (README, AUDIT*, WAVE*, FIXES*, VERIFY*,
W7-READY, PROGRESS*, PARKED, TOOLCALLING_COMPARISON, COMMANDCODE_SETUP, protocol.md, CHANGES.md)
+ 41 loose scripts (analyze_*, final_*, probe_*, tuned_*, walkforward_*, wave*.sh …, 27 of them
now in `scripts/archive/`) + `docs/` (5 files) + `agent_bullwhip/` + `tests/` + `hpc-umbrella/` +
`results/` + `tools/`.

**After:**

```
agent-bullwhip-replication/
├── README.md              ← new entry point (short statement + protocol rules + links)
├── CHANGES.md             ← running change log (cleanup entry appended 2026-10-07)
├── CLEANUP-REPORT.md      ← you are here
├── docs/
│   ├── README.md          ← navigation: layout, naming, how to run, results flow
│   ├── STATE.md           ← wave table 1–7, verdicts, OPEN ISSUES
│   ├── ARTIFACTS.md       ← manifest of untracked-but-kept artifacts
│   ├── waves/             ← WAVE5.md, WAVE6_SPEC.md, WAVE7.md, WAVE7_SPEC.md, W7-READY.md, protocol.md
│   ├── audits/            ← AUDIT.md…AUDIT5.md, AUDIT-RUNTIME-2026-10-01.md, WAVE7-AUDIT-1/2.md
│   ├── fixes/             ← FIXES-2026-10-01.md, FIXES2-2026-10-01.md
│   ├── verify/            ← VERIFY-2026-10-01.md, VERIFY-W7-EVAL.md
│   ├── missions/          ← PROGRESS.md, PROGRESS_DESKTOP.md (wave-6 mission logs)
│   └── archive/           ← README_v1..v4, README_ITERATION_LOG, PARKED.md, COMMANDCODE_SETUP.md,
│                            TOOLCALLING_COMPARISON.md, harness-fault-handling.md
├── scripts/               ← LIVE tooling (12 scripts + README.md; see §3)
│   └── archive/           ← 27 LEGACY one-off scripts + README.md
├── agent_bullwhip/  tests/  hpc-umbrella/  results/  tools/   (unchanged structure)
```

Counts: **68 files moved** (64 via `git mv` — 27 docs + 37 scripts; 4 untracked files re-homed
by plain move: PROGRESS.md, PROGRESS_DESKTOP.md, noisy_eval_arms.sh, run_suite_chunked.sh),
**67 binaries untracked**, **0 deleted**.

## 2. Inventory decision table (LIVE vs LEGACY)

Full rationale in mission `inventory.md`. Summary:

| class | rule | examples |
|---|---|---|
| LIVE → docs/waves/ | current wave docs/specs + the protocol | WAVE5/6/7, WAVE7_SPEC, W7-READY, protocol.md |
| audits → docs/audits/ | adversarial audit docs | AUDIT.md…AUDIT5.md, AUDIT-RUNTIME-2026-10-01.md, WAVE7-AUDIT-1/2.md |
| fixes/verify → docs/fixes/, docs/verify/ | dated infra notes | FIXES-*, VERIFY-* |
| missions → docs/missions/ | historical mission logs (wave 6) | PROGRESS.md, PROGRESS_DESKTOP.md |
| archive → docs/archive/ | superseded / one-off, kept for archaeology | README_v1..v3, README_v4 (pre-cleanup full README), README_ITERATION_LOG, PARKED.md (superseded 2026-09-30), TOOLCALLING_COMPARISON.md, COMMANDCODE_SETUP.md, harness-fault-handling.md |
| LIVE → scripts/ | still used by the current workflow | wave7_analysis.py, wave7_floors.py, bench_rollout.py, walkforward_floor.py, tuned_floor.py, walkforward_paths.py, paired_analysis.py, noisy_arm.sh, noisy_eval_arms.sh, run_suite_chunked.sh, probe_permutation_invariance.py, probe_decision_readout.py |
| LEGACY → scripts/archive/ | one-off from a past wave; kept, not deleted | final_*.py, wave2*.sh, probe_*.py, analyze_*traces*.py, gap/analysis/find/inspect/list_models/run_model_comparison/wave5_report/oos_floor/tuned_floor_chaos/resume/run_all/collect_tool_traces/extract_deltas/compare_jev_vs_text |

Classification signals used: last-commit date vs current wave (7), presence in cluster kit
(`hpc-umbrella/` OPERATOR/config/sbatch references ⇒ LIVE), references from current wave docs,
and whether the produced results are in the surviving evidence sets.

## 3. Move map (all moves, group → destination)

### Docs (29 moves: 27 git-mv + 2 plain)
- `protocol.md`, `WAVE5.md`, `WAVE6_SPEC.md`, `WAVE7.md`, `WAVE7_SPEC.md`, `W7-READY.md` → `docs/waves/`
- `AUDIT.md`, `AUDIT2.md`, `AUDIT3.md`, `AUDIT4.md`, `AUDIT5.md`, `AUDIT-RUNTIME-2026-10-01.md`,
  `WAVE7-AUDIT-1.md`, `WAVE7-AUDIT-2.md` → `docs/audits/`
- `FIXES-2026-10-01.md`, `FIXES2-2026-10-01.md` → `docs/fixes/`
- `VERIFY-2026-10-01.md`, `VERIFY-W7-EVAL.md` → `docs/verify/`
- `README.md` (pre-cleanup full text) → `docs/archive/README_v4-2026-10-07.md` (kept for the
  surviving tables; new short README written at root)
- `docs/README_v1.md`, `docs/README_v2.md`, `docs/README_v3.md`, `docs/README_ITERATION_LOG.md` →
  `docs/archive/` (already in docs/, re-homed)
- `docs/harness-fault-handling.md` → `docs/archive/`
- `PARKED.md`, `COMMANDCODE_SETUP.md`, `TOOLCALLING_COMPARISON.md` → `docs/archive/`
- `PROGRESS.md`, `PROGRESS_DESKTOP.md` → `docs/missions/` (plain move; untracked → now tracked)
- new: `docs/README.md`, `docs/STATE.md`, `docs/ARTIFACTS.md`, `scripts/README.md`,
  `scripts/archive/README.md`

### Scripts (39 moves: 37 git-mv + 2 plain)
- LIVE → `scripts/`: `walkforward_floor.py`, `tuned_floor.py`, `walkforward_paths.py`,
  `paired_analysis.py`, `wave7_analysis.py`, `wave7_floors.py`, `bench_rollout.py`,
  `noisy_arm.sh`, `probe_permutation_invariance.py`, `probe_decision_readout.py`
- LEGACY → `scripts/archive/` (27): `analysis_table.py`, `analyze_jev_traces.py`,
  `analyze_tool_traces.py`, `analyze_toolagent.py`, `collect_tool_traces.py`,
  `compare_jev_vs_text.py`, `extract_deltas.py`, `final_checks.py`, `final_leakfree.py`,
  `final_noisy.py`, `final_val2.py`, `find_traces.py`, `gap_analysis.py`, `inspect_traces.py`,
  `list_models.py`, `oos_floor.py`, `probe_jev.py`, `probe_one_shot.py`,
  `probe_tool_models.py`, `probe_tool_models_light.py`, `resume.sh`, `run_all.sh`,
  `run_model_comparison.py`, `tuned_floor_chaos.py`, `wave2.sh`, `wave2b.sh`, `wave5_report.py`
- plain moves (untracked → now tracked): `noisy_eval_arms.sh`, `run_suite_chunked.sh` →
  `scripts/`

### Reference fixes (every location that mentioned a moved name)
- `hpc-umbrella/OPERATOR.md`, `config.sh` (comment), `cpu-harness.sbatch`, `gpu-pilot.sbatch`,
  `gpu-wave7-eval.sbatch` — script invocations rewritten to repo-relative `scripts/…` (these
  `cd ${REPO_DIR}` before running, so bare names would break); docs refs to
  `../scripts/…` relative to the kit dir.
- All wave/audit/fix/verify/mission docs updated: doc-name references → `../…/NAME.md`
  relative paths; script references → `../../scripts/…` / `../../scripts/archive/…`;
  kit refs (`config.sh`, `stage.sh`, `verify_pull.py`, `OPERATOR.md`, `eval-a0-report.md`,
  `push_results.sh`…) → `../../hpc-umbrella/…`.
- `agent_bullwhip/*.py` module docstrings/comments referencing `WAVE*_SPEC.md`, `AUDIT*.md`,
  `grpo*.py`, `collect_tool_traces.py` → corrected relative paths (comment-only; no behavior).
- `tests/test_demand_sharing.py` docstring: `tuned_floor_chaos.py` → `../scripts/archive/…`.
- `POSTS.md`, post drafts, `.env.local`, `CHANGES.md` history entries, and the archived README
  snapshots (v1–v4, iteration log) were intentionally **not** rewritten (post drafts never
  committed; CHANGES is a historical log; snapshots are archaeology — a label in the new root
  README points at them).

## 4. Binary untracking table

| what | files | size | on disk? | tracked readout evidence kept |
|---|---|---|---|---|
| `results/wave7_{sft,rlcd}_seed{0,1,2}/checkpoint-*/` (optimizer.pt, tokenizer.json, adapter_model.safetensors, merges.txt, vocab.json, configs) | 66 | 261.8 MB | yes (unchanged) | `results/eval/`, `results/wave7_labels/`, `results/wave7_logs/`, `wave7_floors.json` stay tracked |
| `hpc-umbrella/umbrella-results/results/grpo_a0_base/checkpoint-299/adapter_model.safetensors` | 1 | 9.2 MB | yes | `results/grpo_a0_base/state.json`, `train_log.jsonl` stay tracked |
| **total** | **67** | **≈271 MB** | all on disk | — |

`.gitignore` extended: `results/**/checkpoint*/`, `*.pt`, `optimizer*.pt`, `*.safetensors`.
sha256 for every untracked file: `captures/untracked-shas.txt` in the mission dir (also
referenced from `docs/ARTIFACTS.md`). Recover: rerun the train/eval jobs from
`hpc-umbrella/` (documented) or `git show` of the pre-cleanup commits that tracked them.

### umbrella-results tracked-vs-untracked numbers
- on disk: 338 MB, 204 files · tracked: 98 files, ~30 MB (campaign logs, eval-a0-report, bench,
  sweep-check, wave7 text readouts) · untracked: 106 files (checkpoint dirs, wave7_labels JSONL,
  wave7_logs). Big regenerable binaries → untracked; text results → kept. The git-based results
  transfer is untouched: `push_results.sh` (git add -f + commit + push) still works verbatim;
  `verify_pull.py hpc-umbrella/umbrella-results` → **VERDICT: COMPLETE** (all 18 expected runs
  with final artifacts); `pull_results.sh`/`stage.sh` syntax-OK and path-stable.

## 5. Test parity + smoke evidence

| check | before | after | verdict |
|---|---|---|---|
| `uv run python -m pytest -q` | 326 passed, 15 skipped (24.06 s) | 326 passed, 15 skipped (24.55 s) | **identical** |
| `uv run pytest -q` (bare) | 2 skipped, 20 collection errors (`agent_bullwhip` not importable — poetry-style pyproject; pre-existing) | same | unchanged, documented |
| `python -c "import agent_bullwhip"` | OK | OK | pass |
| CLI `python -m agent_bullwhip.interleaved_runner --help` | OK | OK | pass |
| `scripts/wave7_analysis.py` re-run on existing eval set | committed `results/wave7_report.json` | regenerated stats **byte-identical** to committed | pass |
| cluster kit `bash -n` (all *.sh) | — | all OK | pass |

Captures: mission dir `captures/tests-before.txt`, `captures/tests-after.txt` (+ full logs).

## 6. Inconsistencies found and NOT fixed (per constraint 3)

1. **`uv run pytest -q` fails at collection** (20 errors, `agent_bullwhip` not importable).
   pyproject.toml is poetry-style with no `[project]`/`[tool.uv]` table, so bare `uv run pytest`
   does not put the package on `sys.path`. Pre-existing; docs now say `python -m pytest`.
   Recommendation: add a `[tool.uv]`/packaging table or accept the documented command.
2. **CHANGES.md 2026-08-29 entry says 19 one-off scripts were archived to `scratch/`, but
   `scratch/` was deleted in commit `99a6c6e` the same day.** Those scripts survive only in git
   history (`git show 99a6c6e^:scratch/…`). Not fixed (no on-disk copy to move; history entry
   left as written).
3. **`verify_pull.py` documented invocation is against the repo root, but the grpo trees live
   under `hpc-umbrella/umbrella-results/results/`** — root invocation reports MISSING for the
   11 campaign configs; the umbrella-results invocation is COMPLETE. OPERATOR.md line 300's
   example is stale; not changed (kit doc, path-verbatim rule).
4. Root README (pre-cleanup) said "53 tests pass"; the real suite is 326 passed / 15 skipped
   (W7-READY already records 326). Fixed by the rewrite, noted here for the record.

## 7. Fresh-reader test (using ONLY docs/README.md + docs/STATE.md)

| question | answer found in | correct? |
|---|---|---|
| where are the 7 wave docs? | docs/README.md §4 → `docs/waves/` (WAVE5.md, WAVE6_SPEC.md, WAVE7.md, WAVE7_SPEC.md, W7-READY.md; waves 1–3 fold into audits/README history) | yes |
| where are the audits? | docs/README.md §4 → `docs/audits/` | yes |
| where do results live + which are the supported sets? | docs/README.md §5 table (results/noisy_leakfree, chaotic/wild_seeded, jev_*, local-eval, eval, wave7_*, wave7_floors.json) | yes |
| how do I run tests / eval / analysis / training / cluster? | docs/README.md §6 (HOW TO RUN) + §7 (hub↔cluster git rule); cluster: `hpc-umbrella/OPERATOR.md` | yes |
| what did each wave test and what's the verdict? | docs/STATE.md wave table 1–7 + current verdicts | yes |
| what are the open issues? | docs/STATE.md OPEN ISSUES (wave-6 free-form lane validity; Clef-style arm next steps) | yes |
| what is untracked but kept, and how to recover it? | docs/ARTIFACTS.md (reachable from docs/README.md §5) | yes |
| what is this project and its protocol? | root README.md (entry) + protocol rules §2 + docs/STATE.md | yes |

## 8. Remaining known messiness + recommendations

- `results/` still 370 MB of mixed evidence/archaeology; waves 1–5 sets are untracked-but-kept
  (documented in ARTIFACTS.md §3). A future mission could archive pre-fix archaeology dirs under
  a clearly-labeled `results/archive/` — but do not delete them.
- `docs/archive/README_v4-2026-10-07.md` duplicates pre-cleanup README content (kept
  intentionally; the wave docs + results are the ground truth).
- `POSTS.md` / post drafts remain at root, untracked (repo rule). Nothing resembling them was
  committed.
- The pending wave-6/7 experiments (docs/STATE.md OPEN ISSUES) are untouched.
- Recommendation: after this push, any cluster push should pass **text readouts only**
  (`push_results.sh` with results/eval, summaries, logs) — checkpoint dirs stay local/scratch;
  ARTIFACTS.md documents why.

## 9. Summary

- Files moved: **68** (64 via `git mv` — 27 docs + 37 scripts; 4 untracked re-homed by plain move).
- Binaries untracked: **67 files ≈ 271 MB** removed from the tracked set (kept on disk;
  sha256-manifested).
- Tests: 326 passed / 15 skipped before and after (identical). CLI + wave-7 analysis re-run OK.
- Commits: see git log (`[verified]`-prefixed where the suite was green). Push state: see
  `git status` / remote ref at the end of the mission.