# Agent Bullwhip — LLM Agents in the Beer Game, Audited

Replication + extension of **arXiv:2605.17036**, *Reliability and Effectiveness of Autonomous AI
Agents in Supply Chain Management* (Long, Simchi-Levi, Zhu, Su, Calmon & Calmon; Harvard/MIT/
Georgia Tech). LLM agents manage a four-echelon MIT Beer Game. The paper's question: are
autonomous AI agents reliable supply-chain decision-makers? Its fix: GRPO RL post-training. Our
question: does cheap inference-time structure work instead? Our answer took three audits to
arrive at — because the first three answers we produced were wrong. The mistakes are ours,
documented in full; the surviving numbers are the ones that survived.

**The surviving claim, in one line:** a 1970s inventory formula that tunes itself from 18 weeks of
history beats a state-of-the-art LLM agent in every environment — mild, chaotic, wild demand —
at 3–4 orders of magnitude less compute. The LLM's apparent edge was information leakage and an
untuned baseline. The intelligence in the "AI agent" was the formula all along. Every contested
word of that sentence is grounded; the walk-forward number is half in-sample and we do not paper
over that.

This repo spans **7 waves**: API-LLM arms (1–3), a tool-calling arm (4), a decision-native model
(5), local GRPO training (6), and decision-native training — soft-target SFT + RLCD (7). The
latest wave's verdict is **interface positive, training null-to-negative** (closed-choice readout
kills the rogue tail; outcome-label training at 0.6B made policy worse).

---

## The research protocol (binding — read before writing any claim here)

1. **Pre-register.** Commit the design, gates, controls, and null clause *before* running it
   (`*_SPEC.md` files; changes get logged in the spec, never silently).
2. **Every number needs its on-disk source.** No number goes into a doc without the
   `results/` artifact it came from. Readouts/JSONL/logs are tracked; checkpoints are not.
   `VERIFY-*.md` docs recompute headlined numbers from the raw artifacts independently.
3. **Audit before publish.** Results are published only after an adversarial self-audit
   (`AUDIT*.md`). Audits are the content — dead claims are documented, not deleted.
4. **Honest negatives are the deliverable.** A null result is a result. "Mechanism did not
   transfer at our budget" is published as cleanly as a win would be.
5. **Hindsight is published, not hidden.** Tuned-on-a-suffix floors, half-in-sample cost
   columns, retailer lookahead, letter-luck controls — state exactly what each number does and
   does not mean.
6. **Results travel through git, both directions** (hub ↔ cluster). rsync is not used; commits
   record which job produced what.

Full protocol history: `docs/waves/protocol.md`. Read the audits before trusting any headline: `docs/audits/AUDIT.md`,
`docs/audits/AUDIT2.md`, `docs/audits/AUDIT3.md` (the mistakes), `docs/audits/AUDIT4.md`/`docs/audits/AUDIT5.md` (wave 5), `WAVE7-AUDIT-*.md`.

---

## Navigate

| you want | go to |
|---|---|
| **how this repo is organized, where things live, how to run anything** | [`docs/README.md`](docs/README.md) |
| **where the project stands: wave table, verdicts, open issues** | [`docs/STATE.md`](docs/STATE.md) |
| change log (what ran when, why) | [`CHANGES.md`](CHANGES.md) |
| the cluster run kit (wave-6/7 training, TU/e Umbrella) | `hpc-umbrella/OPERATOR.md` |
| the surviving data | `results/` (index in [`docs/README.md`](docs/README.md) §5) |
| untracked-but-kept artifacts (checkpoints, big binaries) | [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) |

## Run (three commands that matter)

```bash
uv run python -m pytest -q            # 379 passed, 16 skipped (after WAVE6-FIX, 2026-10-07)
uv run python scripts/wave7_analysis.py     # wave-7 gate readout → results/wave7_report.md
uv run python scripts/walkforward_floor.py  # walk-forward + oracle floor columns ($0, no API)
```

> `uv run pytest` alone fails at collection (poetry-style pyproject, package not on path) — use
> `python -m pytest`. LLM arms need `FREEINFERENCE_API_KEY`; training runs happen on the cluster
> via `hpc-umbrella/`.

---

*Older full-length READMEs (v1–v3, iteration log, and the v4 pre-cleanup text with the complete
surviving tables) are archived under [`docs/archive/`](docs/archive/) — kept, not deleted. The
surviving tables also remain the on-disk ground truth in `results/` and `CHANGES.md`.*