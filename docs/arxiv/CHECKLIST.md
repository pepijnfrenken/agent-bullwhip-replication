# arXiv submission checklist

Built against tag `arxiv-submission-1` (cut from `6993227`). Nothing here is submitted; the upload is Pino's hand on the key.

## Files

| file | role |
|---|---|
| `paper.pdf` | main text, 7 pages |
| `paper-supplement.pdf` | methods supplement S1–S7, 13 pages |
| `paper.tex`, `paper-supplement.tex` | sources, so the package is rebuildable |
| `metadata.json` | title, abstract, authors, categories, license, reproducibility pointer |
| `SHA256SUMS` | integrity check for the above |

`refs.bib` is absent: the bibliography is inline `\begin{thebibliography}` in `paper.tex`, so there is
no separate bib file to ship. Do not add one to satisfy a habit.

## Metadata to paste into the arXiv form

- **Title**: Post-Tuning Beats Its Own Baseline and Still Loses to a Tuned Rule: LLM Agents from 0.6B to 8B on a Simulated Beer Game
- **Authors**: Pepijn Frenken
- **Primary category**: `cs.AI`; cross-list `cs.LG`
- **Comments**: `7 pages main text, 13 pages methods supplement. Pre-registered adaptation study of
  arXiv:2605.17036v4. The protocol, gates, stop-loss and run ledger are the product; artifacts are
  committed at tag arxiv-submission-1.`
- **License**: CC BY 4.0 for the paper, MIT for the code. Do not describe the code license as
  covering the reference paper's material.
- **Reproducibility pointer**: `https://github.com/pepijnfrenken/agent-bullwhip-replication/tree/arxiv-submission-1`
  — **public, verified 2026-11-25.** `gh repo view` reports `visibility: PUBLIC`, and an
  unauthenticated fetch returns 200 for the repo, the tag tree and the raw PDF. The pointer is
  safe to paste into a public field. The repository now carries `SECURITY.md` with the scan result
  and the one residual disclosure: the cluster account number is redacted from the tree but remains
  recoverable from historical commits, which were not rewritten because the paper cites their shas.

## Decisions already made, and the cost of each

1. **October primary, November as evidence.** Path (a) from the rebuild decision. Body keeps October
   numbers and verdicts; the November rebuild appears in Results, S6 and S7 with its own figures.
   Cost: the October 4B seed-1 figure and its 30/30 claim stay document-recorded, and the paper now
   says so in the abstract rather than at paragraph eleven.
2. **8B reported as one arm, not a rung.** The only pre-registered paired pass is the October
   `1e-5` seed-1 survivor. The selected `5e-5` rate is band-level favourable and explicitly
   unestablished under G1. Cost: the rung reads weaker than the earlier draft, which is the point.
3. **Supplement ships as a second PDF, not appended.** Cost: the reviewer has to open a second file;
   benefit: the main text stays a self-contained 13 pages and the supplement's 13 pages do not count against it.
4. **Supplementaries carry the losses.** S7 names the corrupt transfer, the adapters that did not
   survive, and the third November leg that died with its sandbox. Cost: it is uncomfortable to
   publish. Benefit: nobody can claim the paper hid a ninetieth-percent result.

## Before upload, check these five things

1. `sha256sum -c SHA256SUMS` in this directory passes.
2. No absolute local paths, tokens, or notebook URLs in `metadata.json` or either `.tex`.
3. The abstract in `metadata.json` matches the abstract in `paper.pdf` word-for-word after LaTeX
   stripping. It was generated from the same source, so a mismatch means the source moved.
4. The tag exists on the remote (`git ls-remote --tags origin | grep arxiv-submission-1`) **and** the
   repository reads without credentials (`curl -o /dev/null -w '%{http_code}'` must return 200). It
   returned 404 while private and returns 200 since the flip on 2026-11-25. Publishing had a real
   consequence: it publishes every committed result row, Slurm log and wave memo, and leaves out only
   the LoRA adapters. Make it public deliberately or ship the pointer as "available on request".
5. `CONTACT-NOTE-v0.md` is a separate decision. Sending it to the reference authors is not part of the
   submission.

## What is deliberately not in this package

- No figure files: the paper has no figures, only tables, which are in the `.tex`.
- No run artifacts: the reproduction path is the git repository, not an appendix bundle.
- No cover letter. arXiv takes none.
