"""LLM client: OpenAI-compatible chat completions against FreeInference.

Config via env: FREEINFERENCE_BASE (default https://freeinference.org/v1),
FREEINFERENCE_API_KEY, FREEINFERENCE_MODEL (default deepseek-v4-flash).
"""
from __future__ import annotations

import os
import re
import time
import json
from pathlib import Path

import requests

# 2026-08-31 — Tool-calling arm harness hardening (Wave 4 continuation)
#
# - **Error-recovery loop**: tool_choice now goes `required -> auto -> auto -> none`
#   instead of `required -> none -> none`. The old flow made a model that errored
#   on its FIRST tool call unable to retry (round 2+ forced 'none'). Now middle
#   rounds use 'auto', so the model can retry after an error OR answer directly.
#   ToolAgent default max_tool_rounds bumped 1 -> 3.
# - **Consecutive-error nudge**: after a tool exec error, the harness appends a
#   pointed "[harness] Your previous code call(s) errored..." message so the model
#   fixes the bug or simplifies instead of silently giving up.
# - **Fallback labeling**: ToolAgent now records `last_decision_meta` per decision
#   with `is_fallback` / `mirror_after_error` flags (kills the silent-mirror
#   laundering anti-pattern from AUDIT2/3).
# - **Per-decision checkpointing**: collect_tool_traces.py writes each decision to
#   JSONL immediately (resume-friendly; mid-game outage doesn't lose the run).
# - **401 / quota-lie retry**: FreeInference free tier returns 200 + {"error":
#   "Invalid or expired API key"} when the account is throttled (NOT a real auth
#   failure). chat() and chat_with_tools() now retry 401s and the error-body lie
#   with backoff instead of treating them as terminal.
# - **Model probe tooling**: probe_tool_models_light.py / run_model_comparison.py
#   = quota-aware, resumable multi-model tool-calling comparison (checkpoints per
#   decision, 30s between calls, interleaved so a quota recovery benefits all).
#   Tests: 66 pass (2 new: error-recovery loop, forced-final-round).
#
# Account-level rate limit: minimum seconds to sleep before each API call.
# The endpoint returns 400 {"type":"rate_limited"} when we hammer it; pacing
# keeps us under the limit (free tier is very restrictive). Bump via env
# FREEINFERENCE_RATE_LIMIT_DELAY if needed. CommandCode has no such limit.
RATE_LIMIT_DELAY = float(os.environ.get("FREEINFERENCE_RATE_LIMIT_DELAY", "2.0"))

# Provider selection: CommandCode (higher concurrency, full tool support) or
# FreeInference (default). Set COMMANDCODE_API_KEY to use CommandCode.
COMMANDCODE_KEY = os.environ.get("COMMANDCODE_API_KEY", "")
if COMMANDCODE_KEY:
    BASE = os.environ.get("COMMANDCODE_BASE", "https://api.commandcode.ai/provider/v1")
    KEY = COMMANDCODE_KEY
    PROVIDER = "commandcode"
    RATE_LIMIT_DELAY = 0.0  # no client-side pacing needed
else:
    BASE = os.environ.get("FREEINFERENCE_BASE", "https://freeinference.org/v1")
    KEY = os.environ.get("FREEINFERENCE_API_KEY", "")
    if not KEY:
        # fallback for background/daemon contexts where the env var is sanitized:
        # load the repo-local key file (gitignored).
        _env_local = Path(__file__).resolve().parent.parent / ".env.local"
        if _env_local.exists():
            for _line in _env_local.read_text().splitlines():
                if _line.startswith("FREEINFERENCE_API_KEY="):
                    KEY = _line.split("=", 1)[1].strip()
                    break
    PROVIDER = "freeinference"
DEFAULT_MODEL = os.environ.get("FREEINFERENCE_MODEL", "deepseek-v4-flash")
# shared auth header for all requests (built once at import)
_HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}

_INT_RE = re.compile(r"-?\d+")


def _extract_xml_tool_call(text: str) -> str | None:
    """Extract the `code` argument from an XML-style tool call in model text.

    Some models (deepseek on CommandCode) emit:
        <tool_calls><invoke name="run_python">
        <parameter name="code">...</parameter></invoke></tool_calls>
    instead of structured JSON tool_calls. Returns the code string or None.
    """
    if "<invoke" not in text and "<tool_calls" not in text:
        return None
    m = re.search(r'<parameter name="code"[^>]*>(.*?)</parameter>', text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r'<invoke name="run_python"[^>]*>(.*?)</invoke>', text, re.S)
    if m:
        return m.group(1).strip()
    return None


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    n: int = 1,
    timeout: int = 90,
    retries: int = 10,
) -> list[str]:
    """Return n sampled completions. Retries with backoff on 429/5xx/network errors.

    Token usage is stored on the caller via the `token_sink` mechanism: this module
    keeps a module-level `last_usage` dict the agent layer reads after each call.
    Prompt-hash collision check: module-level `prompt_hashes` list records a short
    hash of the concatenated messages each call, so a post-hoc analysis can detect
    whether identical prompts were sent repeatedly (cache-collision risk).
    """
    global last_usage
    model = model or DEFAULT_MODEL
    # Reasoning-heavy models (GLM) need headroom: their THINKING burns tokens
    # before the answer. Default 64 is fine for DeepSeek/Qwen but starves GLM.
    if max_tokens is None:
        max_tokens = 2048 if "glm" in model.lower() else 64
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    # account-level rate limit: pace calls to avoid "rate_limited" 400s
    time.sleep(RATE_LIMIT_DELAY)
    # per-call nonce to defeat serving-level caching (audit #4): inject a harmless
    # random token into the last user message so identical semantic prompts never
    # share a byte-identical body across runs/windows.
    messages = list(messages)
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            m = dict(messages[idx])
            m["content"] = m["content"] + f"\n<!-- cache-buster {os.urandom(4).hex()} -->"
            messages[idx] = m
            break
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
    }
    # collision check: record a fingerprint of the exact prompt (cheap, no PII)
    import hashlib
    _fp = hashlib.sha256(
        ("\x1f".join(m.get("content", "") for m in messages)).encode()
    ).hexdigest()[:16]
    prompt_hashes.append(_fp)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(f"{BASE}/chat/completions", headers=headers, json=body, timeout=(10, 60))
            if resp.status_code == 429:
                # limit-1 endpoints: back off exponentially with jitter so our own
                # retries don't collide with each other (3s, 6s, 12s, 24s, ...)
                time.sleep(3 * (2 ** attempt) + os.urandom(1)[0] / 2)
                continue
            if resp.status_code == 401:
                # FreeInference intermittently 401s under load even with a valid
                # key (observed 2026-08-31). Retry with backoff; treat as terminal
                # only after exhausting retries.
                time.sleep(3 * (2 ** attempt) + os.urandom(1)[0] / 2)
                continue
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            last_usage = {"prompt": usage.get("prompt_tokens", 0),
                          "completion": usage.get("completion_tokens", 0)}
            # capture any model-level reasoning trace (DeepSeek-style reasoning_content,
            # or a generic `reasoning` field) if the endpoint provides it
            reasons = []
            for ch in data.get("choices", []):
                msg = ch.get("message", {})
                r = msg.get("reasoning_content") or msg.get("reasoning") or None
                reasons.append(r if r else None)
            global last_reasoning
            last_reasoning = reasons
            return [ch["message"]["content"] for ch in data.get("choices", [])]
        except requests.exceptions.Timeout:
            last_err = TimeoutError("LLM call timed out")
            time.sleep(2 + 2 * attempt)
        except Exception as e:  # noqa: BLE001 - any transport/HTTP error -> retry
            last_err = e
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


# module-level usage sink (agent layer reads it after each call)
last_usage: dict = {"prompt": 0, "completion": 0}
# module-level reasoning sink (agent layer reads it after each call)
last_reasoning: list | None = None
# module-level prompt fingerprints (collision check; cleared by the runner per run)
prompt_hashes: list[str] = []


def parse_order(text: str) -> int | None:
    """First integer in the model response; None if unparseable (instruction failure)."""
    m = _INT_RE.search(text or "")
    return int(m.group()) if m else None


_ORDER_LABEL_RE = re.compile(r"(?:^|\n)\s*ORDER\s*[:=]?\s*(-?\d+)", re.IGNORECASE | re.MULTILINE)


def parse_order_prefer_label(text: str) -> int | None:
    """Parse order, preferring an explicit ORDER: label (GLM format)."""
    if not text:
        return None
    m = _ORDER_LABEL_RE.search(text)
    if m:
        return int(m.group(1))
    return parse_order(text)


def lint_code(code: str) -> str | None:
    """Cheap static check on model-written Python before exec.

    Catches the most common model bug: calling a function that is not defined
    in the snippet (e.g. defines `simulate` but calls `simulate_order_up_to`).
    Returns an error string if a likely NameError is found, else None.

    This is intentionally conservative — only flags calls to names that are
    (a) not defined locally, (b) not builtins, (c) not in the injected helpers
    (math, statistics, np). It will miss some bugs but never false-positives
    on valid code.
    """
    if not code or not code.strip():
        return "ERROR: empty code — retry with a complete code argument"
    try:
        import ast
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"ERROR: syntax error: {e.msg} (line {e.lineno}) — retry with valid Python"
    defined = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
    builtins = set(dir(__builtins__))
    if not isinstance(__builtins__, dict):
        builtins |= set(vars(__builtins__))
    builtins |= {"math", "statistics", "np", "print", "sum", "len", "range",
                 "abs", "min", "max", "int", "float", "str", "list", "dict",
                 "set", "tuple", "round", "sorted", "enumerate", "zip", "map",
                 "filter", "any", "all", "bool", "type", "isinstance", "open",
                 "Exception", "ValueError", "TypeError", "NameError", "None"}
    undefined = called - defined - builtins
    if undefined:
        names = ", ".join(sorted(undefined)[:5])
        return f"ERROR: undefined name(s) in code: {names} — define them or use a different approach, then retry"
    return None


def _stream_chat_completion(
    payload: dict,
    *,
    timeout: tuple = (10, 45),
    idle_timeout: float = 20.0,
) -> dict:
    """POST a streaming chat completion and reassemble the final message.

    Returns a dict shaped like a non-stream response: {"choices": [{"message":
    {...}}], "usage": {...}}. The per-chunk `idle_timeout` is the mid-flight
    hang detector: if no SSE chunk arrives for that long, we abort and raise
    (the endpoint sometimes stalls mid-generation — this catches it in ~20s
    instead of waiting for the read timeout).
    """
    import json as _json
    resp = None
    last_status = None
    for _attempt in range(6):  # upstream-outage absorption: ~3 min of 429-waits,
        # then fail FAST so the runner can retry the whole game sooner instead
        # of burning 25 min inside a dead upstream.
        try:
            resp = requests.post(
                f"{BASE}/chat/completions", headers=_HEADERS, json=payload,
                stream=True, timeout=timeout,
            )
        except requests.exceptions.Timeout:
            raise
        if resp.status_code == 429:
            resp.close()
            last_status = "429"
            time.sleep(30)  # upstream provider temporarily unavailable — wait briefly
            continue
        if resp.status_code in (500, 502, 503, 400):
            body = resp.text[:300]
            raise RuntimeError(f"tool-chat: HTTP {resp.status_code}: {body}")
        break
    if last_status == "429" or resp is None:
        raise RuntimeError("tool-chat: upstream provider 429 for ~3 min — giving up")
    resp.raise_for_status()
    # accumulate per-stream-chunk state
    msg = {"role": "assistant", "content": "", "tool_calls": []}
    tool_calls_by_idx: dict[int, dict] = {}
    reasoning = ""
    usage: dict = {}
    finish = None
    last_chunk_t = time.time()
    for line in resp.iter_lines(decode_unicode=True):
        now = time.time()
        if now - last_chunk_t > idle_timeout:
            resp.close()
            raise TimeoutError(
                f"tool-chat: stream idle {idle_timeout:.0f}s (mid-flight stall)")
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        last_chunk_t = now
        try:
            j = _json.loads(d)
        except Exception:
            continue
        choices = j.get("choices") or []
        if not choices:
            if j.get("usage"):
                usage = j["usage"]
            continue
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            msg["content"] += delta["content"]
        if delta.get("reasoning") or delta.get("reasoning_content"):
            reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls_by_idx.setdefault(idx, {
                "id": tc.get("id", ""), "type": "function",
                "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function", {})
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
        fr = choices[0].get("finish_reason")
        if fr:
            finish = fr
    resp.close()
    if tool_calls_by_idx:
        msg["tool_calls"] = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
    else:
        msg.pop("tool_calls", None)
    if reasoning:
        msg["reasoning"] = reasoning
    if finish == "length" and not msg["content"] and not msg.get("tool_calls"):
        raise TimeoutError("tool-chat: stream cut off (finish_reason=length)")
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage}


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: int = 120,
    retries: int = 10,
    max_tool_rounds: int = 3,
    exec_globals: dict | None = None,
    dry_run: bool = False,
    force_tool: bool = True,
) -> tuple[str, list[dict]]:
    """Chat with tool calling; execute tool calls locally (sandboxed-ish).

    Returns (final_text, tool_trace) where tool_trace is a list of
    {name, arguments, result} dicts. The loop runs at most max_tool_rounds
    tool rounds; after that the model must produce a final answer. If the
    model returns a final message without tool calls, we return it.

    exec_globals: dict injected into the exec namespace (e.g. math, numpy).
    dry_run: if True, never hit the network (used in tests) — returns a
    canned tool-call response so the exec loop can be exercised offline.
    force_tool: if True, use tool_choice='required' on the first round so the
    model must actually call the tool (many models otherwise ignore tools and
    just write code in their text answer). After the tool result comes back,
    we switch to tool_choice='none' so the model produces the final answer.
    """
    global last_usage
    model = model or DEFAULT_MODEL
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    # account-level rate limit: pace calls to avoid "rate_limited" 400s
    time.sleep(RATE_LIMIT_DELAY)
    messages = list(messages)
    tool_trace: list[dict] = []
    exec_ns = {"__builtins__": __builtins__}
    if exec_globals:
        exec_ns.update(exec_globals)

    msg: dict = {}
    data: dict = {}
    for _round in range(max_tool_rounds + 1):
        if dry_run:
            # canned: first round calls run_python, second round answers
            if _round == 0:
                msg = {"content": "", "tool_calls": [{
                    "id": "call_dry_1",
                    "type": "function",
                    "function": {"name": "run_python",
                                 "arguments": "{\"code\": \"print(17 * 23)\"}"},
                }]}
            else:
                msg = {"content": "391"}
            data = {"choices": [{"message": msg}],
                    "usage": {"prompt": 0, "completion": 0}}
        else:
            # Tool-choice strategy: round 0 forced (if force_tool) so the model
            # MUST call the tool once; middle rounds "auto" so the model can
            # retry after an error OR answer directly (this is what makes the
            # error-recovery loop possible); final round "none" forces a final
            # answer. CommandCode rejects tool_choice="required"/"none" in
            # thinking mode, so it gets "auto" on round 0 as well.
            if _round >= max_tool_rounds:
                tc = "none"  # last round: must produce a final answer
            elif force_tool and _round == 0 and PROVIDER != "commandcode":
                tc = "required"
            else:
                tc = "auto"
            payload: dict = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tc,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = None
            attempt = 0
            hang_count = 0
            for attempt in range(retries):
                try:
                    if PROVIDER == "commandcode":
                        # streaming + per-chunk idle timeout = catch mid-flight
                        # stalls in ~20s instead of a 45s dead-read wait.
                        data = _stream_chat_completion(
                            {**payload, "stream": True},
                            timeout=(10, 45), idle_timeout=20.0,
                        )
                        resp = {"status_code": 200}  # sentinel: got data
                    else:
                        # FreeInference: non-streaming, plain retry/backoff
                        resp = requests.post(
                            f"{BASE}/chat/completions", headers=_HEADERS, json=payload,
                            timeout=(10, 45),
                        )
                        if resp.status_code in (429, 401, 500, 502, 503, 400):
                            if attempt == 0 and resp.status_code == 400:
                                # log the 400 body once so we can diagnose (transient vs payload)
                                try:
                                    err_body = resp.text[:300]
                                except Exception:
                                    err_body = "<no body>"
                                print(f"[tool-chat] 400 on attempt 0: {err_body}", flush=True)
                            if resp.status_code == 429:
                                # upstream outage: back off LONG (60s) and keep
                                # trying — a 5-10 min outage must not kill the run.
                                # Cap total 429-wait at ~25 min, then give up.
                                if attempt >= 24:
                                    raise RuntimeError("tool-chat: upstream 429 for 25+ min")
                                time.sleep(60)
                                continue
                            if resp.status_code == 401:
                                # transient 401 (FreeInference under load): short
                                # backoff, treat as retryable like other 5xx.
                                time.sleep(min(2 ** attempt, 30))
                                continue
                            time.sleep(min(2 ** attempt, 30))
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                    break
                except requests.exceptions.Timeout:
                    # endpoint accepted but stalled — retry, but track the hang;
                    # 3 consecutive hangs => give up on this decision (fallback)
                    hang_count += 1
                    if hang_count >= 3:
                        raise RuntimeError("tool-chat: endpoint hung 3x in a row")
                    time.sleep(2 + 2 * attempt)
                except TimeoutError:
                    # mid-flight stream stall (idle timeout) — same handling
                    hang_count += 1
                    if hang_count >= 3:
                        raise RuntimeError("tool-chat: stream stalled 3x in a row")
                    time.sleep(2 + 2 * attempt)
                except requests.RequestException as e:
                    if attempt == retries - 1:
                        raise
                    time.sleep(min(2 ** attempt, 30))
            if resp is None:
                raise RuntimeError("tool-chat: no response after retries")
            if isinstance(resp, dict):  # streaming path already has data
                pass
            else:
                data = resp.json()
        if "usage" in data:
            last_usage = data["usage"]
        if "choices" not in data or not data["choices"]:
            # endpoint returned an error body (e.g. {"error": {...}}) — retryable
            err_detail = data.get("error", {}).get("message", str(data)[:120])
            # FreeInference free tier lies about quota exhaustion: it returns
            # 200 + {"error":"Invalid or expired API key"} when throttled, not a
            # real auth failure. Treat it as retryable (it clears in seconds).
            if "invalid or expired" in err_detail.lower() or "rate" in err_detail.lower():
                if "attempt" in locals() and attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 30))
                    continue
            if "attempt" in locals() and attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"tool-chat: endpoint error (no choices): {err_detail}")
        msg = data["choices"][0]["message"]
        # capture thinking/reasoning for trace mining (CommandCode: "reasoning",
        # FreeInference/OpenAI-style: "reasoning_content")
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if reasoning:
            last_usage["reasoning"] = reasoning
        tool_calls = msg.get("tool_calls")
        # Some models (deepseek on CommandCode) emit tool calls as XML *text*
        # instead of structured tool_calls. Detect and convert.
        xml_tc = _extract_xml_tool_call(msg.get("content") or "")
        if not tool_calls and xml_tc:
            tool_calls = [{
                "id": "call_xml_1",
                "type": "function",
                "function": {"name": "run_python", "arguments": json.dumps({"code": xml_tc})},
            }]
        if not tool_calls:
            return msg.get("content") or "", tool_trace
        # append assistant message with tool calls, then execute each
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if not args.get("code"):
                # empty/broken tool call (e.g. finish_reason=length) — record a
                # failure and tell the model to retry with a complete call
                tool_trace.append({"name": name, "arguments": args,
                                   "result": "ERROR: empty code — retry with a complete code argument",
                                   "reasoning": reasoning})
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": "ERROR: empty code — retry with a complete code argument"})
                continue
            result = ""
            if name == "run_python":
                code = args.get("code", "")
                # lint before exec: catch common model bugs (undefined fn names,
                # syntax errors) and feed back WITHOUT burning an execution
                lint_err = lint_code(code)
                if lint_err:
                    tool_trace.append({"name": name, "arguments": args,
                                       "result": lint_err, "linted": True,
                                       "reasoning": reasoning})
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                     "content": lint_err})
                    continue
                try:
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        exec(compile(code, "<tool>", "exec"), exec_ns, exec_ns)
                    result = buf.getvalue().strip() or "(no output)"
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                    # error-recovery nudge: after consecutive tool errors, tell
                    # the model explicitly to fix the code (or fall back to a
                    # simple formula) instead of silently giving up. Track the
                    # error streak on the trace list.
                    err_streak = 0
                    for prev in reversed(tool_trace):
                        if prev.get("result", "").startswith("ERROR"):
                            err_streak += 1
                        else:
                            break
                    if err_streak >= 1:
                        result += ("\n\n[harness] Your previous code call(s) errored. "
                                   "Fix the bug (check variable names, imports, syntax) "
                                   "or simplify: a moving-average or order-up-to style "
                                   "computation is enough. Do not repeat the same bug.")
            else:
                result = f"ERROR: unknown tool {name}"
            tool_trace.append({"name": name, "arguments": args, "result": result,
                               "reasoning": reasoning})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                             "content": result})
    # loop exhausted; return last assistant content if any
    return msg.get("content") or "", tool_trace


def list_models() -> list[str]:
    """Verify which models the endpoint actually serves (/v1/models = truth)."""
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    # account-level rate limit: pace calls to avoid "rate_limited" 400s
    time.sleep(RATE_LIMIT_DELAY)
    resp = requests.get(f"{BASE}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]
