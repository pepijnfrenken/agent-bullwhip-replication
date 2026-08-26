# Post drafts — agent-bullwhip-replication (final, audit-surviving)

Both follow the socials KB: number-led hook, Discord-text tone, no guru voice,
link in FIRST COMMENT (not the body), one honest negative, real question.

---

## X post (short, punchy)

I replicated the "agent bullwhip" paper (LLM supply-chain agents, Beer Game)
on a free API.

Plain LLM agents are chaos machines: cost CV ≈ 4.97, ~10-50x worse than the
paper's 0.13-0.46.

A deterministic order-up-to anchor + confidence gate tames them completely:
4,113 / CV 0.073, within 12% of the floor.

Then I audited my own result. The "win" was mostly the wrapper, not the LLM.
And under noisy demand, the LLM only looked better because it had an
information leak: upstream tiers saw the same-week order of the tier below.

Fix the leak, tune the formula (1-minute grid search), and the formula beats
the best LLM agent by 10-20% on cost AND consistency.

The "agent" is the formula wearing an LLM costume. The intelligence was in
the wrapper, the leak, and the hyperparameters — not the model.

Full writeup with both audits, all results, and the leak-free re-run:
[link in first comment]

Anyone else found LLM "agent" wins that evaporate once you check what
information the model was actually given?

---

## LinkedIn post (adapted, recruiter-friendly, link in first comment)

I replicated a Harvard/MIT paper on LLM agents managing a supply chain
(the Beer Game) — and then audited my own result hard enough to kill it.

What I found:

• Plain LLM agents are unreliable: running the same scenario 30x, cost
  varies with CV ≈ 4.97. The paper's reliability concern replicates — worse.

• Adding a deterministic inventory formula (order-up-to) as a guardrail
  tames them completely: 4,113 mean cost, CV 0.073 — within 12% of the
  theoretical floor.

• But the formula, not the LLM, was doing the work. A wrapper-only run
  scored exactly the floor. The LLM inside added cost, not value.

• Under uncertain demand, the LLM looked like it adapted better — until I
  found it was reading the current week's downstream orders (an info leak
  the formula didn't have). Fix the leak, tune the formula's 2 parameters,
  and the formula beats the best LLM agent by 10-20% on cost and 2x on
  consistency.

The honest takeaway: when information is equal, a tuned formula beats the
LLM agent. The "intelligence" in many AI agents is the wrapper and the
data they're handed — not the model.

Both full audits, the leak fix, and every result are in the repo:
[link in first comment]

For people building LLM agents: do you audit what information the model
gets vs. your baseline? Or just compare final numbers?

---
