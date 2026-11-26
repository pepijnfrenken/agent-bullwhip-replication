# CONTACT NOTE v1 - draft to the authors of arXiv:2605.17036

Status: draft, not sent. Sending is Pino's call. v0 is superseded and kept only for the record.

## Who to email, and why that order

v4 prints no email addresses and names no corresponding author (checked: `pdftotext` over the
whole PDF, zero matches for an address pattern). Addresses below come from institutional pages,
verified 2026-11-26, not from memory.

| order | person | address | source of address | why |
|---|---|---|---|---|
| To | Carol Xuan Long | carol_long@g.harvard.edu | Harvard SEAS person page | first author; was a Harvard applied-math PhD student, now ETH AI Center postdoc; the person who most plausibly ran the experiment and the one most likely to answer |
| Cc | David Simchi-Levi | dslevi@mit.edu | MIT CEE and ORC faculty pages | senior author; the name that decides whether a reply is worth writing. Note: `dslevi@`, not `dsimchi@` |
| optional Cc | Flavio P. Calmon | flavio@seas.harvard.edu | Harvard CRCS page and his CV | senior author on the theory side |

Do not send to all six. A six-recipient cold email with a technical claim reads as noise and
gets triaged as noise.

Mechanics that matter more than the wording:

- Send from one address and keep it: the one that will also be the arXiv corresponding email.
- **Decision taken (Pino, 2026-11-26): attach the PDF, and keep the repo as a link.** The
  attachment is `Frenken-2026-agent-bullwhip-adaptation.pdf`, a renamed copy of
  `docs/arxiv/paper.pdf` (83,322 bytes, sha256 `d21aab58…f8f2e`-series verified byte-identical), because
  `paper.pdf` in an inbox says nothing about whose work it is. Mitigations that keep the attachment
  from costing deliverability: plain text, no images, exactly one link (the repo), the PDF as the only
  attachment, and sending from a domain with SPF and DKIM set. This overrides my earlier no-attachment
  advice at Pino's direction; the argument against it stands on record and is not being erased.
- The repository is already public: opened 2026-11-25, re-verified 2026-11-26 (repo 200, tag PDF 200
  unauthenticated). Nothing needs opening before sending. Accept the consequence: the replication has
  been publicly readable for a day, so a reader who goes looking for the title can find it before
  arXiv has it.
- Tuesday to Thursday, mid-afternoon CET: Zurich is the same hour, Boston is morning.
- One follow-up at 10-14 days, quoting the three questions, then stop. Do not chase a third time.
- A second, shorter email the day the preprint goes up. That one gets replies; cold ones often
  do not, and silence is not a verdict on the work.


| v0 said | why it is wrong now | source |
|---|---|---|
| subject: "negative at 8B" | one 8B arm passes G1: 14,594 vs 89,633 control, 29/30 | `docs/paper/paper.tex` §4, Table 3 |
| 8B "3.0x the trained-4B reference, slightly worse than untrained" | the October survivor reads 0.163 of the untrained 8B control | same |
| 1.7B "5e-5 trips the collapse gate outright" | single reads tripped, no arm was stopped; adverse on all three seeds by cost | §4, §6 |
| 4B "~1.4-1.6x our walk-forward floor" | 1.75x at seed 0, 3.36x at seed 1, against the rule's 5,346 | §4, Table 2 |
| addressed "Carol, Huangyuan" on its own | the names are right (Carol Xuan Long, Huangyuan Su, now verified against the arXiv listing), but v0 had no addresses and implied a two-recipient blast; v1 sends to one, copies two | arXiv listing, institutional pages |
| omitted the tuned-rule result | it is the paper's central claim and the most useful thing to tell them | §4 |

---

Subject: Pre-registered four-size adaptation study of your Beer-Game post-training claim

Dear Dr Long and colleagues,

I am Pepijn Frenken, an independent researcher in Eindhoven (the TU/e Umbrella L4 lane provided
compute only). I have spent the last several months on a pre-registered adaptation study of the claim in your v4
(arXiv:2605.17036, 21 September 2026), and I want to write to you before it goes to arXiv, because
you are the only people who can tell me whether I have read your protocol correctly.

Two findings, one of which is about your paper rather than about your method.

The first is that your own rule is never compared against. Equation 5 (the exponentially smoothed
forecast, following Chen et al. 2000b) and equation 6 (the order-up-to target) define a tuned
order-up-to policy, and the Appendix C.1 model carries it analytically, but I could not find any
result anywhere in v4 in which that tuned rule is scored as a cost baseline against the LLM arms.
If I have missed one, that is the first thing I want to be told. I ran it as the reference: two parameters fitted on weeks 1-18 of a 36-week
game, then played frozen. It costs 5,346 in median across 30 demand paths. Nothing I trained at any
size comes within 1.75x of it, and the closest arm is 1.75x only at its best seed.

The second is the ladder, run as your recipe adapted to my harness (exact expected-cost gradient
instead of sampled-K, frozen-anchor KL, closed-choice letter readout). At 4B post-training does
beat its own untrained baseline: 9,346 against 21,578 at seed 0 (30/30 paired paths) and 17,958 at
seed 1 (25/30, and marginal against my 15% bar). At 1.7B it is adverse on all three seeds, roughly
2.4x to 8.2x the untrained cost. At 0.6B it is negative, but that rung used a different method
(SFT and RLCD rather than the RL objective), so I do not read it as evidence about your recipe. At
8B, one of four arms passes (14,594 against 89,633, 29/30), and that arm is the one my frozen
selection rule did not pick, against a control whose readout is poorly conditioned.

Three questions, none of which is a request for approval:

1. Is my reading of eqs. 5-6 right, and is it right that v4 contains no tuned-rule comparison? If
   it does and I missed it, I would rather correct my paper than publish the claim that it isn't
   there.
2. Did you see seed variance at 4B? Mine is 1.92x between two seeds of the same arm, which makes
   me suspect a single-seed result is fragile at this scale in ways that are not visible in v4.
3. Do per-run artifacts exist behind the reported numbers? Mine are committed and public, and I
   would rather compare row by row than result by result.

The paper, the pre-registration, the gates and every per-run file are public at
https://github.com/pepijnfrenken/agent-bullwhip-replication (tag `arxiv-submission-1`), including a
section listing where I deviated from my own protocol. If you tell me I have misread something,
that is a correction I will make and credit, and it is much cheaper before submission than after.
[[Pino: set the submission window here, or delete this sentence.  Do not send a date you have not
decided on.]]

I have not asked for your endorsement and I am not asking now. I am asking whether I am wrong.

Best regards,
Pepijn Frenken
Independent researcher, Eindhoven
https://github.com/pepijnfrenken/agent-bullwhip-replication