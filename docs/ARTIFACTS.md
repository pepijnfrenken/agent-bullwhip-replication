# ARTIFACTS — untracked-but-kept artifacts (2026-10-07)

Manifest of files that exist on disk (and in the git-visible history where noted) but are
**not tracked** by git, with size, sha256, why they are untracked, and how to regenerate or
recover them. Rule of the repo: *nothing may become unrecoverable or undocumented.* Anything in
this file that is missing or damaged on disk is a data-loss event — report it, do not silently
regenerate (the whole point of the untracking was that these are regenerable with known inputs).

> Context: `results/` is gitignored by default. The wave-7 checkpoint dirs and the
> `grpo_a0_base` adapter used to be force-tracked (`push_results.sh` does `git add -f`);
> the 2026-10-07 cleanup untracked them (`git rm --cached`, files untouched on disk) and
> extended `.gitignore` (`results/**/checkpoint*/`, `*.pt`, `*.safetensors`) so they do not
> come back. Text readouts (eval JSONL, summaries, labels, logs, floors JSON) **stay tracked** —
> they are the evidence the research protocol requires.

## 1. Wave-7 training checkpoints — untracked 2026-10-07 (66 files, 261.8 MB)

Directories `results/wave7_sft_seed{0,1,2}/checkpoint-119/` and
`results/wave7_rlcd_seed{0,1,2}/checkpoint-39/` (11 files each):
`adapter_config.json`, `adapter_model.safetensors`, `added_tokens.json`, `chat_template.jinja`,
`merges.txt`, `optimizer.pt`, `README.md`, `special_tokens_map.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`.

- **Why untracked:** regenerable artifacts of the training runs (job 390955); the branch-point
  evidence is the `train_log.jsonl` + `state.json` + eval readouts in `results/eval/`, all
  tracked. The weights add 260+ MB to every clone for no provenance gain.
- **Recover — from the cluster (canonical):** re-run the train job via
  `hpc-umbrella/gpu-wave7-train.sbatch` (same pins => same names), or pull from any cluster
  checkout that still has them via `push_results.sh`/git.
- **Recover — from git history (pre-cleanup):** these exact files are in the repo's git history
  up to commit `020f43d`/`2a8ff87` era (they were tracked before the 2026-10-07 cleanup, e.g.
  `git show <pre-cleanup-commit>:results/wave7_sft_seed0/checkpoint-119/tokenizer.json`).
- **sha256:** stored in `captures/untracked-shas.txt` in the cleanup mission dir
  (`/home/pino/missions/bullwhip-cleanup/`), one line per file (path, sha256, bytes), generated
  2026-10-07 with `sha256sum` before/at untrack time.

| entry | size | sha256 (first 16) |
|---|---|---|
| `results/wave7_sft_seed{0,1,2}/checkpoint-119/tokenizer.json` (×3) | 11 MB each | see manifest file |
| `results/wave7_sft_seed{0,1,2}/checkpoint-119/optimizer.pt` (×3) | 18 MB each | see manifest file |
| `results/wave7_sft_seed{0,1,2}/checkpoint-119/adapter_model.safetensors` (×3) | 8.8 MB each | see manifest file |
| `results/wave7_rlcd_seed{0,1,2}/checkpoint-39/tokenizer.json` (×3) | 11 MB each | see manifest file |
| `results/wave7_rlcd_seed{0,1,2}/checkpoint-39/optimizer.pt` (×3) | 18 MB each | see manifest file |
| `results/wave7_rlcd_seed{0,1,2}/checkpoint-39/adapter_model.safetensors` (×3) | 8.8 MB each | see manifest file |
| (remaining 46 small config/text files) | — | see manifest file |

## 2. Wave-6 GRPO adapters in umbrella-results — untracked 2026-10-07 (1 file, 9.2 MB)

- `hpc-umbrella/umbrella-results/results/grpo_a0_base/checkpoint-299/adapter_model.safetensors`
  — 9,189,904 bytes, sha256 in the mission captures.
- **Why untracked:** same rule — regenerable training artifact; the train log + state.json for
  `grpo_a0_base` remain tracked (`results/grpo_a0_base/state.json`, `train_log.jsonl`).
- **Recover:** re-run the campaign job (`hpc-umbrella/gpu-campaign.sbatch`, `campaign.tsv`
  row `a0_base`), or from git history pre-cleanup, or from any cluster checkout.

## 3. Pre-existing untracked artifacts (not touched by this cleanup)

These were never tracked (existing `results/` gitignore); listed here so the manifest is
complete and nothing is "undocumented":

| path | size | why untracked | recover |
|---|---|---|---|
| `results/` waves 1–5 evidence dirs (`noisy_leakfree/`, `chaotic_seeded/`, `wild_seeded/`, `jev_protocol/`, `jev_controls_*/`, `jev_noisy20/`, `toolagent_*`, `wrapper_only`, `interleaved_val*`, …) | ~80 MB | raw run traces; README/audits reference them; regenerable only by re-running the respective arms | re-run the interleaved runner / Jev arms with the documented seeds; some (pre-fix) sets are unreproducible by design and exist only as evidence |
| `hpc-umbrella/umbrella-results/results/grpo_a*_*/` checkpoint dirs (18 runs) | ~300 MB | regenerable training output; the run evidence (`train_log.jsonl`, `state.json`, eval readouts) is tracked | re-run campaign / pull from cluster |
| `.venv/` | 241 MB | toolchain, not evidence | `uv sync` / `pip install -r requirements.txt` |
| `uv.lock` | 52 B | empty stub (poetry-style pyproject: uv cannot resolve a `[project]` table; `uv run` works via requirements) | regenerate with `uv run` |
| `POSTS.md`, `POST_wave5_repost.md` | — | **post drafts — repo rule: never committed** | local drafts only |

## 3b. WAVE6-FIX probe outputs (staged 2026-10-07; not yet produced)

Produced by `hpc-umbrella/gpu-w6fix.sbatch` (WAVE6-FIX_SPEC §8.1); come back through `hpc-umbrella/push_results.sh`.

| path | tracked? | content |
|---|---|---|
| `results/grpo_w6fix_{a0,a0b,a1,a1_seed2,a2,a2_seed2}/` — `train_log.jsonl`, `state.json`, `run_info.json`, `completions_step{0,10,…}.jsonl` | tracked (`git add -f`) | per-step telemetry (`contract_parse_rate` …), run flags (`contract_version: w6fix-1`), full completion dumps |
| `results/grpo_w6fix_*/checkpoint-*/` | **untracked** (`.gitignore` `results/**/checkpoint*/`) | LoRA adapters; regenerable by re-running |
| `results/w6fix_report.{md,json}` | tracked | generated by `python3 scripts/w6fix_analysis.py --out results/w6fix_report.md --json results/w6fix_report.json` |
| `results/w6fix_logs/w6fix-<jobid>_<task>.log` | tracked | Slurm logs (provenance) |
| `tests/fixtures/w6fix_collect_legacy_golden.json` | tracked | pre-change `collect_step` goldens + first-prompt SHAs (V1) |

## 3c. WAVE8 artifacts (code + hub pre-flight committed 2026-10-07; training/eval not yet run)

| path | tracked? | content |
|---|---|---|
| `results/wave8/grid_w8_g26.json`, `grid_w8_g26b.json` | tracked (`git add -f`) | generated choice sets (`scripts/wave8_grid.py`): values, labels, instruction, prompt SHA (string renderer + `apply_chat_template`), label token ids |
| `results/wave8/f0_w8_g26_prereg.json` (+ `.J.json`) | tracked | pre-registered `w8_g26` F0 read: **FAIL** (see WAVE8_SPEC changes log); reference-grid J rows for reproduction |
| `results/wave8/f0.json` (+ `.J.json`), `f1.json`, `gates.json` | tracked | F0 (revised `w8_g26b`, fresh seeds), F1 teacher check, S_nom / oracle summaries; F2/F3 are filled by the cluster pre-flight |
| `results/wave8/S_nom.jsonl`, `S_nom_oracle.json` | tracked | the fixed 1,024-state sample and its oracle labels (J, p, argmin) |
| `results/wave8_<arm>_<lr>_seed<S>/` (`round_state.json`, `collapse_reads.jsonl`, `round<r>/` working files) and `results/wave8_<arm>_<lr>_seed<S>_round<r>/` (`train_log.jsonl`, `state.json`, `pi_sft.json`) | train logs tracked; `checkpoint-*` **untracked**; per-round state/label files regenerable | R1/R2+RL runs |
| `results/wave8_val/w8-val-*.jsonl` | tracked | curriculum-validation evals (lr selection, band 9500..9529) |
| `results/eval/w8-*.jsonl`, `*.summary.json`, `w8-*-perm.json`; `results/wave8/<arm>-S_nom.json`, `lr_selection*.json`, `gates_<rung>.json` | tracked | protocol evals with per-decision `traces` + `readout` key (W8 §7.2), gate reads |
| `results/wave8_report.{md,json}` | tracked | `scripts/wave8_analysis.py` readout |
| `results/wave8_gguf/` — `manifest.json`, `*.summary.json`, equivalence json | tracked | **`*.gguf` and the merged HF dir are untracked** (`.gitignore`); sha256 in `manifest.json` |

## 4. What to do after a clean clone

1. `git clone` — you get code, docs, tests, and the tracked wave-7 readouts under `results/eval`,
   `results/wave7_labels`, `results/wave7_logs`, `results/wave7_floors.json`.
2. Checkpoints/adapters are NOT in the clone. If you need the trained weights, copy them from
   the hub checkout (they live on disk here) or re-run training on the cluster.
3. `uv run python -m pytest -q` to verify the tree (481 passed / 20 skipped after WAVE6-FIX + WAVE8 staging, 2026-10-07; 326 / 15 before).

## 5. Integrity check (cheap, recommended on recovery)

```bash
# compare any recovered file against the manifest in the cleanup mission dir
sha256sum -c <(grep <path> /home/pino/missions/bullwhip-cleanup/captures/untracked-shas.txt)
```