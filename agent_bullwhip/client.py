"""LLM client: OpenAI-compatible chat completions against FreeInference.

Config via env: FREEINFERENCE_BASE (default https://freeinference.org/v1),
FREEINFERENCE_API_KEY, FREEINFERENCE_MODEL (default deepseek-v4-flash).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

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
DEFAULT_MODEL = os.environ.get("FREEINFERENCE_MODEL", "deepseek-v4-flash")

_INT_RE = re.compile(r"-?\d+")


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
            resp = requests.post(f"{BASE}/chat/completions", headers=headers, json=body, timeout=timeout)
            if resp.status_code == 429:
                # limit-1 endpoints: back off exponentially with jitter so our own
                # retries don't collide with each other (3s, 6s, 12s, 24s, ...)
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


_ORDER_LABEL_RE = re.compile(r"\bORDER\s*[:\-]?\s*(-?\d+)", re.IGNORECASE)


def parse_order_prefer_label(text: str) -> int | None:
    """Parse order, preferring an explicit ORDER: label (GLM format)."""
    if not text:
        return None
    m = _ORDER_LABEL_RE.search(text)
    if m:
        return int(m.group(1))
    return parse_order(text)


def list_models() -> list[str]:
    """Verify which models the endpoint actually serves (/v1/models = truth)."""
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    resp = requests.get(f"{BASE}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]
