# Generalizable Decision Playbook (v1 — domain-agnostic)
# Injected into the LLM prompt as context.
# These are TRANSFERABLE control principles — they apply to any dynamic system
# with feedback, delays, and uncertainty (supply chains, markets, queues,
# multi-agent coordination). No domain-specific constants or formulas here.

## Principle 1: Distinguish signal from noise (don't over-react to transients)
- **Trigger:** a single large observation / spike in the input you're responding to.
- **Rule:** a one-off jump is usually a level shift or noise, not a trend. Respond
  proportionally and confirm persistence before escalating your action. Smooth
  noisy inputs (moving average) before acting on them.
- **Why:** over-reacting to a single point creates oscillation that propagates
  through any coupled system (the bullwhip is the canonical example). Patience is
  cheap; over-correction compounds.

## Principle 2: Cover your obligations before optimizing
- **Trigger:** you have unmet obligations (backlog, debt, pending commitments).
- **Rule:** resolve the obligation first, then act on new information. But do not
  over-correct: cap your response so you don't create a second wave of instability
  upstream. Order of operations: obligations > new signals > optimization.
- **Why:** ignoring obligations while chasing new signals leaves the system
  permanently behind. But over-committing to clear an obligation causes overshoot
  — the classic "chase the tail" failure.

## Principle 3: Account for delays — what you did earlier arrives later
- **Trigger:** you have actions in flight (orders placed, work in progress,
  pipeline, lead time).
- **Rule:** in-flight actions do NOT help you NOW — they arrive after the delay.
  Never treat in-flight work as current capacity. Compute your *effective*
  position = what you have + what's coming − what you owe, and act on the gap.
- **Why:** ignoring delay is the #1 source of oscillation in any delayed-feedback
  system. You either double-order (forgetting what's coming) or under-act
  (treating future arrivals as present relief). Both compound.

## Principle 4: Know when you don't know — and fall back to safety
- **Trigger:** uncertainty is high, information is ambiguous, or your confidence
  is low.
- **Rule:** do not guess with an extreme action. Fall back to a smooth,
  conservative, proportional response — the "do no harm" default. Extreme actions
  under uncertainty are how small errors become large disasters.
- **Why:** the expected cost of a wild guess under uncertainty is almost always
  worse than a smooth default. Reliability comes from knowing your limits.

## Principle 5: Smooth, don't chase
- **Trigger:** the input you're responding to is noisy / oscillating.
- **Rule:** use an exponentially-weighted (or moving) average of recent inputs as
  your estimate, not the raw last value. Act on the trend, not the wiggle.
- **Why:** raw values over-react to noise; smoothing damps the oscillation that
  otherwise propagates through coupled systems and amplifies upstream.

## Principle 6: Structure beats willpower (make the safe path the default)
- **Trigger:** you are designing a system or a repeated decision process.
- **Rule:** prefer *structural* constraints (caps, fallbacks, defaults) over
  relying on moment-to-moment judgment. Design the environment so the safe action
  is the path of least resistance.
- **Why:** judgment degrades under pressure; structure does not. The most reliable
  systems are those where the default behavior is already correct.

---

## Usage notes
- These principles are **domain-agnostic**: they transfer to any system with
  feedback, delay, and uncertainty. The beer-game rules in the domain KB are
  specializations of these.
- The agent should **map its observed state onto these principles** and state in
  its reasoning which principle(s) it applied.
