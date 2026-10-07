# scripts/archive/ — legacy one-off scripts

One-off tools from past waves, kept on disk and in git history (moved here 2026-10-07 during the
repo cleanup; moved with `git mv`, so history is intact). None are part of the current workflow;
do not rely on them for new claims. The audits/WAVE docs still cite some of them — those
references now point here.

- `final_*.py` — pre-audit run drivers (sequential-protocol era; killed by AUDIT1–3).
- `probe_*.py` — model/config probes (wave 4–5 exploration).
- `analyze_*`, `collect_tool_traces.py`, `compare_jev_vs_text.py`, `extract_deltas.py` —
  trace/analysis tools for the tool-calling and Jev arms.
- `wave2.sh`, `wave2b.sh`, `run_all.sh`, `resume.sh` — wave-2-era shell drivers.
- `oos_floor.py`, `tuned_floor_chaos.py` — audit-era floor variants (superseded by
  `walkforward_paths.py`).
- `list_models.py`, `run_model_comparison.py`, `analysis_table.py`, `gap_analysis.py`,
  `find_traces.py`, `inspect_traces.py`, `wave5_report.py` — one-off analysis.

Historic note: CHANGES.md (2026-08-29) says 19 older one-off scripts were archived to `scratch/`;
that directory was removed by commit `99a6c6e` the same day, so those 19 survive only in git
history (`git show 99a6c6e^:scratch/…`). See `CLEANUP-REPORT.md`.