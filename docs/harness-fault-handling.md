# How Harnesses Deal with Faulty Issues — pi-mono, OMP, prime-agent, herdr, and the bullwhip replication

**Date:** 2026-08-30 · **Method:** evidence-first — (1) a fault-event census of every local agent trace (`~/.pi/agent/sessions`, `~/.omp/agent/sessions`; script `trace_scan.py`), (2) reading the retry/fallback/backoff code in the pi-ai + pi-coding-agent installs (`~/.nvm/.../@earendil-works/...`), the OMP harness source (`~/.bun/install/global/node_modules/@oh-my-pi/`), prime-agent (`/home/pino/projects/prime-agent/`), and herdr's integration, (3) the bullwhip paper's own audit trail (AUDIT.md / AUDIT2.md / AUDIT3.md, `collect_tool_traces.py`, `analyze_tool_traces.py`). Everything below cites what was actually read; nothing is inferred from memory.

Evidence flags: **VERIFIED** (read in code/doc directly) · **CORRELATIONAL** (observed in traces, mechanism best-effort) · **SMALL-N** (tiny sample) · **UNVERIFIED** (could not locate locally).

---

## 1. Measured fault counts — the trace census

`trace_scan.py` (evidence artifact, run 2026-08-30T05:34:23Z; output at `/tmp/shortform-clip-research/trace_scan_counts.txt`).

| Metric | pi-mono (`~/.pi/agent/sessions`) | OMP (`~/.omp/agent/sessions`) | Combined |
|---|---|---|---|
| Files / events | 9 / 638 | 838 / 33,687 | 847 / 34,325 |
| `provider_transport_failure` / provider-fault events | 1 | 9 | 10 |
| Assistant messages with `stopReason:"error"` (provider-level aborts) | 3 | 249 | 252 |
| toolResult events | 250 | 10,024 | 10,274 |
| **Tool-call failures (`isError:true`)** | 23 (9.2%) | 594 (5.9%) | 617 (**6.01%**) |
| Tool errors by tool | bash 23 | bash 404, edit 78, read 53, write 10, launch 9, glob 8, eval 7, todo 5, web_search 5, browser 3, grep 3, find 2, hub 2, ask 2, wait 1 | — |
| OMP `session_exit` | — | 821: dispose 621 (normal), sigterm 103 (signal), sighup 44 (signal), **uncaught_exception 38 (fatal)**, exit 15 (process_exit) | — |
| Lines mentioning 429/rate-limit | 24 | 1,495 | 1,519 |
| Lines mentioning backoff/retry | 16 | 825 | 841 |
| Lines mentioning fallback | 10 | 891 | 901 |
| `"LLM call failed after N attempts"` (provider retry exhaustion) | 1 | 156 | 157 |
| Lines mentioning zombie | 0 | 17 | 17 — **all from this session's own grep, not historical evidence** |

Caveats: the pi corpus is tiny (9 sessions; several dirs hold no session files), so pi ratios are indicative only; OMP includes live sessions written mid-scan (the zombie 17 are self-references); marker counts are line-mentions, not unique events; tool-error rate ≈6% of tool calls is the honest number to quote for "how often a tool call fails in practice".

Worst sessions by fault events: icml2026-repro (36), absolver (26), humanify-evader (22), kalshi_trading (20). Failure clustering is project-specific, not global.

## 2. How each layer handles faults

### 2.1 pi-ai providers — transport retries (VERIFIED, `openai-codex-responses.js` + `openai-completions.ts`)

- `DEFAULT_MAX_RETRIES = 0` in the Codex provider — **provider-level retry is opt-in**, not default; `BASE_DELAY_MS = 1000`, `DEFAULT_MAX_RETRY_DELAY_MS = 60_000`.
- Retryable classification (`isRetryableError`): HTTP 429/500/502/503/504 + regex (`rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused`). **Terminal rate limits are not retried** (`GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|quota exceeded|billing`).
- `getRetryAfterDelayMs`: honors `retry-after-ms`, `retry-after` (seconds), and HTTP-date; else exponential `1000 * 2**attempt`, capped at 60 s. Network errors retryable with the same exponential.
- Transport fallback: Codex websocket → SSE on failure, recorded via `appendAssistantMessageDiagnostic("provider_transport_failure", …, {configuredTransport, fallbackTransport, eventsEmitted})` — **this is the event the trace census counts**; the failure becomes visible history, not a silent retry, and the fallback is attempted in-stream.
- OMP openai-completions: `fetchWithRetry` (Retry-After-aware, 408/429/5xx, "maxRetries: 5, i.e. 6 requests"), first-event watchdog + idle-timeout iterator (stalled streams become timeout errors — observed fix for z.ai/GLM keepalive stalls), post-finish terminal grace (2.5 s), `withEmptyCompletionRetry` (empty `finish_reason:stop` responses re-invoked), strict-tools 400 → retry without strict, Copilot-model retry with linear backoff. VERIFIED (`utils/openai-http.ts`, `utils/retry.ts`, `utils/empty-completion-retry.ts`, `providers/openai-completions.ts`).

### 2.2 pi-coding-agent (pi-mono) — session-level auto-retry (VERIFIED, `dist/core/agent-session.js`)

- After an agent run ends in error, `_willRetryAfterAgentEnd` + `_prepareRetry` restart the **entire agent run** (not the HTTP call): `retry.maxRetries` default **3**, `baseDelayMs` default **2000**, delay = `baseDelay * 2^(attempt−1)`, abortable. `auto_retry_start`/`auto_retry_end` events are emitted; the error message is stripped from agent state (kept in session history) before retry.
- Retryability classifier (`_isRetryableError`): one big regex covering overloaded/provider error/rate limit/429/5xx/service unavailable/network/connection lost/websocket closed/fetch failed/premature stream end/HTTP2/timeout/terminated. **Non-retryable: context overflow** (→ compaction instead) and usage-limit/quota errors; auth failures handled separately (`_isConcreteProviderAuthFailure`).
- **Checkpoint/resume**: every event is appended to a per-session JSONL; `--resume`/`--continue`/`/resume` reload a session and restore model, thinking level, and full message state (`sdk.js`: `agent.state.messages = existingSession.messages`); `/import`, `/export`, fork all reuse the same mechanism. Compaction (`/compact`) is the context-overflow path, not a retry.

### 2.3 OMP (the @oh-my-pi harness) — the hardened fork (VERIFIED, source under `~/.bun/install/global/node_modules/@oh-my-pi/`)

Everything in 2.1/2.2 plus account-pool resilience:

- **Live-429 → `blocked` → account recovery** (`pi-coding-agent/src/session/codex-auto-reset.ts`): a real 429 with a parsed unblock timestamp triggers a planner (`trigger: "blocked"`) that tries sibling credentials/accounts, reconciles against refreshed `/wham/usage` reports (which are IP-throttled — "failure right after a 429 is common", so a snapshot may predate the block; the live 429 evidence is treated as authoritative), synthesizes a candidate from `identity` when no report survives, and can auto-redeem banked Codex resets on a schedule (`sweep`). This is the deepest fault layer in the stack: not just retry, but *credential rotation + quota-window planning*.
- **Subagent retry visibility** (`task/render.ts`, `task/types.ts`): a child sleeping between provider retries (e.g., 429 with retry-after) is surfaced as a "retry-blocked" badge — `progress.retryState` — so parents never misread provider backoff as progress.
- **Tool/hub messaging rate limiter** (`agent-messages.ts`): token bucket (capacity 3, refill 1 s) so peer messaging can't self-DoS; `retry after Nms` errors.
- Trace uploads (`agent-traces.ts`): retriable status set `{408, 425, 429, 500, 502, 503, 504}`, Retry-After respected, bounded by a 60 s window — the harness's *own telemetry* gets the same backoff discipline.
- herdr pane integration in OMP ships as an extension; see 2.4.

### 2.4 prime-agent + herdr — externalized state, 429 → `blocked` UI (VERIFIED, `packages/coding-agent/src/core/extensions/builtin/herdr-agent-state.ts`)

- The built-in herdr reporter (Unix socket to the Herdr pane-manager, `HERDR_SOCKET_PATH`/`HERDR_PANE_ID`/`HERDR_ENV=1`) reports `working` / `idle` / **`blocked`** states with a message.
- **Error → hold → blocked:** when the turn's final assistant message has `stopReason:"error"`, the extension holds "working" through a retry grace window (`HERDR_PI_RETRY_GRACE_MS`, default 2500 ms) *instead of second-guessing the agent's retry classifier with its own pattern list* — "if a retry starts, agent_start keeps the pane working; if none does, the hold settles to blocked with the error message." This is the key design choice: the external observer waits out the internal retry decision rather than duplicating it.
- `herdr:blocked` events (refcounted `blockedCount`) let any layer push the pane into blocked with a label; sequence numbers are monotonic across session instances so a *successor* session (new/resume/fork/reload) can't have its idle reports dropped by the pane's per-source seq guard — i.e., the state reporter itself is made resumable.

### 2.5 herdr OMP integration — "maps 429s to blocked"

- The graceful path is 2.4: provider errors (429 included) → retry-grace hold → `blocked` with the error message. The task-badge path (2.3) shows provider-429 sleeps as retry-blocked. Both are OMP fork features; the *codex-auto-reset* layer is the 429-specific recovery. VERIFIED in source; no separate "429 → blocked" hook beyond these was found.

### 2.6 The bullwhip replication — small experiment, strongest lessons (VERIFIED in repo + AUDITs)

The paper's own handling of faulty issues is the best documented local case:

- **Client-side pacing + jittered exponential backoff** (`agent_bullwhip/client.py`): `RATE_LIMIT_DELAY = 2.0 s` between calls (400 `{"type":"rate_limited"}` on the free tier); `chat()` retries=10, on 429 `sleep(3 * 2**attempt + os.urandom(1)[0]/2)` (jittered, 3 s, 6 s, 12 s…), other errors `2 + 2*attempt`; `chat_with_tools()` on 429 sleeps 60 s per attempt with a ~25-min cap then fails fast "so the runner can retry the whole game sooner instead of burning 25 min inside a dead upstream". Also: per-call nonce to defeat serving-side caches, code lint before exec, sandboxed exec, module-level `prompt_hashes` collision hooks. **"Zombie-runner cleanup" could NOT be located in the repo or logs — UNVERIFIED locally**; the jittered backoff IS verified in code.
- **Silent deterministic fallback (the cautionary tale)**: `parse_order` = first integer anywhere (`client.py:103-107`); any unparseable response → `_fallback(ctx)` = **mirror** policy (`agents.py`, `fallback: str = "mirror"`), and failure-fallback decisions write **no trace entry**. Consequences (AUDIT2 §4, AUDIT3 §6): `kb_pointer_verbal` was **21–29% silent mirror fallbacks** (412/1,440 noisy = 28.6%, 324 chaotic = 22.5%, 298 wild = 20.7%; noisy arm also 461/1,440 = **32%**), its reported mean/CV are unlabeled mixtures, and its worst run (11,522 = +85% vs floor) was a fallback cascade (raw orders 35→95→112→124 mounting through tiers). Lesson: **a fallback that is invisible in the output statistics is worse than a failure** — it launders the fault into a number.
- **Gate/override mechanisms that were decorative**: confidence gate fired **13/4,320 = 0.3%** of decisions (AUDIT.md §3; and just 3/1,440 interleaved, AUDIT2), anchor clamp changed 0–2 decisions across windows; verbal gate regex (`cover (the )?backlog|order-?up-?to|…|pipelin|outstanding`) over-triggered — simulated on real traces it was cost-neutral (**−4% / +1% / ±0%**) with **~80% no-op triggers** (84%/74%/76% where the floor itself ordered 0). The "winner" config's real mechanism was the deterministic anchor: wrapper-only ablation (0 chat calls, 0 tokens, exact floor 3,681) reproduced the win; the LLM inside cost +11.7%–34%.
- **Deterministic floor as the ultimate fallback** (the pattern that *worked*): order-up-to exponential-smoothing policy = 3,681 on fixed demand; a tuned (θ,λ) floor beats the gated LLM on demand-matched tests: chaotic perm p = **0.0153** (LOOCV p = 0.0191, Wilcoxon 0.0195), wild p = **0.0038**; even the untuned floor ties the LLM on matched demand. "Structure > raw LLM judgment on this endpoint" is the audit-surviving claim.
- **Audit-verification as a fault layer** (AUDIT.md/2/3 — the meta-layer): every headline was re-derived from disk; statistical re-derivation (permutation/Wilcoxon/bootstrap, n=10–30), demand reconstruction from traces (rescued the broken crossover arm after the **unseeded-demand bug** destroyed per-pass pairing — interleaved_runner.py seeds only `noisy`), confound checks (same-week information leak worth 37–40%; KB text changed mid-matrix), and the protocol-deviation log (retries 3 promised, 6 used). The audits' fixes became repo features: interleaved round-robin runner, truncate-on-start, run-UUIDs, per-run KB snapshots, cache-buster nonces, leak fix + test, tuned-floor baselines.
- Trace classification infrastructure (`collect_tool_traces.py` / `analyze_tool_traces.py`): per-decision records with floor-vs-model delta, error classification (`NameError`…`other_error`), code-pattern tagging, latency — i.e., **faults are first-class data with a taxonomy**, and the taxonomy is what made the fallback-laundering visible.

## 3. Taxonomy — pattern → where → verdict

| Pattern | Where (verified) | Verdict |
|---|---|---|
| Retry with exponential backoff (+jitter) | pi-ai codex provider (opt-in, 60 s cap); OMP openai-completions `fetchWithRetry` (5); bullwhip `client.py` (10, jittered 3·2^n, 60 s/25-min caps) | **Works** — bounded, Retry-After-aware, jitter prevents thundering herd; terminal/usage-limit errors excluded |
| Full-run auto-retry above the HTTP layer | pi-coding-agent session (`maxRetries 3`, `baseDelayMs 2000`, classified regex, retry events, state rollback) | **Works** — retries the *turn*, not the request; correct for semantic failures; stateful (error stripped, history kept) |
| Timeout watchdogs (first-event, idle, post-finish grace) | pi-ai openai-completions iterators (20 s header timeout; keepalive-stall fix) | **Works** — converts infinite stalls into retryable errors; bounds every attempt |
| Transport fallback (websocket→SSE) recorded as diagnostic | pi-ai codex provider (`provider_transport_failure` diagnostic) | **Works** — visible fallback; the census found it in traces (1× in pi sessions) |
| Deterministic floor / anchor fallback | bullwhip wrapper (anchor, order-up-to); gated configs | **Works** — but only honestly when it's *labeled*; the same fallback as a silent mirror (bullwhip) is the #1 failure pattern |
| Silent fallback with no trace entry | bullwhip `_fallback(ctx)` = mirror on parse failure | **Broken / cautionary** — 21–32% invisible decisions; launder aggregate stats; wrote it in AUDIT2/3 as the top integrity issue |
| Gate/override with a threshold | bullwhip confidence gate (0.3% / 3 of 1,440), anchor clamp, verbal-gate regex | **Mostly decorative here** — fired too rarely to be the mechanism; regex gate ~80% no-op; verdict: measure activation before believing a gate |
| Account pooling + quota-window planning (429 → blocked → rotate/redeem) | OMP `codex-auto-reset.ts` (live-429 evidence > stale usage snapshots; sibling creds; auto-redeem) | **Works** — the deepest layer; needs the live-429 timestamp to beat stale telemetry |
| Externalized state with retry-grace hold | prime-agent herdr extension (working→hold→blocked; monotonic seq across session instances) | **Works** — observer defers to the internal retry decision instead of duplicating it |
| Checkpoint/resume | pi session JSONL + `--resume`/`/import`/fork; bullwhip truncate+run-UUID+KB snapshots | **Works** — every event persisted; resume restores model + messages; audit-driven data hygiene |
| Audit/verification layer (re-derivation, statistics, confound checks) | bullwhip AUDIT 1–3 as a repeatable practice; trace classification taxonomy | **Works** — the layer that caught the silent fallbacks, the unseeded-demand bug, and the 16× window drift |
| Deterministic floor beats LLM on matched tests | bullwhip AUDIT3 (p = 0.015/0.019 chaotic, 0.004 wild on matched demand; OOS stable) | **Works** — the strongest evidence that a cheap deterministic fallback is often the correct answer |

## 4. What Pino should steal for any automated pipeline (the steal list)

1. **Label every fallback.** Any time the pipeline substitutes a decision (retry-exhausted, parse failure, provider error), write it into the record (`is_fallback`, `order_used`). Bullwhip's 21–32% silent-mirror problem is the canonical failure: unlabeled fallbacks corrupt all downstream stats and hide systemic faults.
2. **Live-fault evidence beats stale telemetry.** OMP's codex-auto-reset treats the *live 429's unblock timestamp* as authoritative over pre-block usage snapshots. In a clipping pipeline: an upload API's Retry-After header outranks your local rate-limit cache.
3. **Jittered exponential backoff with a hard cap and fail-fast.** 3·2^n + jitter (bullwhip), Retry-After honoring (pi-ai/OMP), total-window cap then *fail the whole job* so a supervisor can retry (bullwhip's "fail fast so the runner retries the whole game"). Never retry inside a dead upstream for 25 min.
4. **Timeout watchdogs at every layer** — first-byte, stream-idle, post-finish grace — converting stalls into retryable errors (OMP's keepalive-stall fix). A "no output for N minutes" alarm is a fault handler, not a nicety.
5. **Deterministic floor as the default action.** Bullwhip's closest thing to a universal law: on matched demands a tuned 1970s formula beat the LLM 9/10 passes (p ≤ 0.019). For clipping: a rule-based caption/segment fallback (FFmpeg + timestamps) should be what publishes when the AI layer is sick — never a blank post, never a silently wrong one.
6. **Checkpoint-resume at every step.** Tail JSONL per stage (uploaded, rendered, posted, claimed) so a re-run resumes, not restarts; truncate-on-start + run-UUID (bullwhip data-integrity fix) so logs never mix invocations.
7. **Externalize state with a retry-grace hold.** Like herdr: report "working" through the retry window, "blocked" only after it settles — one observer, no duplicated classifiers — and make the reporter itself resumable (monotonic seq).
8. **Audit yourself.** Re-derive headline numbers from raw traces, with a taxonomy of error classes; the bullwhip audits found every major bug (silent fallbacks, unseeded demands, window drift, leak) exactly because trace data was first-class and re-derived.
9. **Count the faults.** This doc's census (≈6% tool-call failure rate; 156 provider-retry-exhaustions in OMP traces; 38 fatal session exits) is the baseline a pipeline should keep: if your failure rate moves, you should know before the revenue does.

## Appendix — trace-count table (raw)

Full output: `/tmp/shortform-clip-research/trace_scan_counts.txt`; scanner: `/tmp/shortform-clip-research/trace_scan.py`. Combined: 847 files, 34,325 events, 10 provider-fault events, 252 provider-level aborts (`stopReason=error`), **617 tool-call failures of 10,274 toolResults (6.01%)**, 1,519 lines mentioning 429/rate-limit, 157 mentions of provider retry exhaustion, 821 OMP session exits (38 fatal `uncaught_exception`).

All sources were read-only; the only write was this document. The `trace_scan` numbers are reproducible by re-running the script.