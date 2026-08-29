# Scratch / archaeology

One-off diagnostic, monitoring and trace-mining scripts from the audit process.
Not needed to reproduce the results in the README — kept for the record.

- `check_*.py` — live progress/status checkers for the various interleaved and crossover runs (Audit 1-3 era). One-shot; obsolete once a run finished.
- `mine_traces*.py`, `inspect_trace_struct.py` — trace-mining scripts used to locate where the LLM diverges from the deterministic floor. Post-hoc analysis, not part of the protocol.
- `diag_interleaved.py` — diagnostics for interleaved-run failures.
- `probe_endpoint.py` — direct endpoint probe (no client retries) used to separate 429 throttling from model behavior.
- `postmortem_verbal.py` — one-off post-mortem on `kb_pointer_verbal` gate firing.
- `wave2d.sh` — Wave 2d launch script (generalizable KB configs); superseded by the interleaved protocol.

The scripts that *do* produce the surviving results live in the repo root:
`final_noisy.py`, `final_val2.py`, `tuned_floor_chaos.py`, `noisy_arm.sh`,
`walkforward_floor.py`, `tuned_floor.py` — see README.md "Reproduce".
