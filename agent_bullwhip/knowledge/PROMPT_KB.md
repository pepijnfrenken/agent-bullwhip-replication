# Knowledge base: supply-chain decision playbook (v1 — seeded)
# Injected into the LLM prompt as context. Each entry = one actionable rule.
# Entries are plain markdown; the agent builder injects them verbatim.

## KB entries (each gets a stable ID + trigger conditions)

### kb-1: demand step (the classic beer-game spike)
- **Trigger:** incoming order jumps >= 2x vs the previous week, or your backlog is growing.
- **Rule:** do NOT over-react with a massive order. A single demand step is usually a
  level shift, not a trend. Increase your order by at most ~1.5-2x the *smoothed*
  incoming, and only after confirming the step persists 2+ weeks.
- **Why:** over-reaction to a step is the #1 cause of the bullwhip (paper §5; classic
  beer-game finding). The cost of waiting a week is small; the cost of over-ordering
  is a multi-week backlog AND holding-cost oscillation.

### kb-2: backlog pressure
- **Trigger:** your backlog (unfilled customer orders) is > 0.
- **Rule:** first cover the backlog *before* adding inventory for expected demand.
  Order at least `backlog + smoothed incoming`, but do not multiply blindly: cap the
  order at `2 x smoothed incoming + backlog` to avoid chasing the tail.
- **Why:** backlog is real, but ordering 10x to clear it creates a second wave of
  bullwhip upstream. The optimal policy is order-up-to: q = target - (on-hand +
  outstanding - backlog), where target ≈ (lead time + 1) x forecast.

### kb-3: lead time (shipments take 2 weeks)
- **Trigger:** you placed an order recently but haven't received it yet (outstanding > 0).
- **Rule:** account for what's already in the pipeline. Never order as if the pipeline
  is empty. `effective position = on-hand + outstanding - backlog`. Order only the
  gap to your target.
- **Why:** ignoring the pipeline is the "double ordering" failure mode — you order
  again for demand that your outstanding orders already cover, doubling the wave.

### kb-3b: in-transit is NOT available this week (measured gap, 2026-08-25)
- **Trigger:** you have outstanding orders (in the 2-week pipeline) AND current
  backlog / this-week demand.
- **Rule:** outstanding orders arrive in 2 WEEKS — they do NOT cover this week's
  demand. If your backlog is positive or demand is coming this week, DO NOT zero
  your order just because the pipeline looks full. Order the gap to cover
  `backlog + this-week incoming`, treating outstanding as *future* relief only.
- **Why:** the trace analysis found the #1 failure mode: agents with large
  outstanding orders order 0 ("already in transit"), the 2-week lag means nothing
  arrives now, and backlog compounds into a bullwhip. In-transit is future
  inventory, not current inventory.

### kb-4: low confidence = fall back to the safe deterministic policy
- **Trigger:** you are uncertain (confidence < 0.5) or the situation is ambiguous.
- **Rule:** do not guess. Return the order-up-to / mirror quantity (a smooth,
  proportional response) rather than an extreme number.
- **Why:** the measured baseline failure mode is extreme orders from confident
  hallucinations. A conservative, smooth response always beats a wild guess on cost.

### kb-5: smoothing (exponential)
- **Trigger:** incoming demand is noisy / oscillating week to week.
- **Rule:** use an exponential moving average (alpha ~0.5) of incoming as your demand
  estimate instead of the raw last-week number.
- **Why:** the raw number over-reacts to noise; smoothing kills the oscillation that
  propagates upstream (paper eqs. 3-5 use the same smoothing).

## Gap log (append here as you find decision failures)
<!-- When a decision is confident but wrong, add: `- [date] <trigger> -> <what the
     model missed> -> <kb-id that would have fixed it or NEW>` -->
