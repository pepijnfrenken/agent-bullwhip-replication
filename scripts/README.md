# scripts/ — live tooling

Scripts still used by the current workflow (analysis, floors, cluster benches, eval launchers).
Archive of one-off/legacy scripts from past waves: `archive/`.

| script | what it does |
|---|---|
| `walkforward_floor.py` | walk-forward + oracle floor columns (waves 1–3 surviving tables). No API. |
| `tuned_floor.py` | reproducible (θ,λ) tuned-floor sweep. |
| `walkforward_paths.py` | per-path walk-forward/untuned/oracle floors on protocol seeds (wave 5/6/7 floors). |
| `paired_analysis.py` | per-path paired deltas + t-stats (wave-5 Jev arms). |
| `wave7_analysis.py` | wave-7 gate readout from `results/eval/` → `results/wave7_report.md`/`.json`. |
| `wave7_floors.py` | wave-7 30-pass floors (reads eval artifacts). |
| `bench_rollout.py` | cluster rollout-batch benchmark (µs/generate, tokens/s, speedup) → JSON. |
| `noisy_arm.sh` | wave-6/7 noisy-arm launcher (see `hpc-umbrella/umbrella-results/sweep-363909-check.md`). |
| `noisy_eval_arms.sh` | wave-7 eval launcher (W7-READY). |
| `run_suite_chunked.sh` | run pytest one file per process (exact-stack, 15GB hub constraint). |
| `probe_permutation_invariance.py` | wave-7 permutation-gate probe (VERIFY-W7-EVAL). |
| `probe_decision_readout.py` | stage-1 Qwen3-0.6B letter-logit readout demo (pre-package). |

Run with `uv run python scripts/<name>.py …` (or `.venv/bin/python`).