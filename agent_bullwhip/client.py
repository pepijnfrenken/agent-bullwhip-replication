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
    max_tokens: int = 64,
    n: int = 1,
    timeout: int = 60,
    retries: int = 3,
) -> list[str]:
    """Return n sampled completions. Retries with backoff on 429/5xx/network errors.

    Token usage is stored on the caller via the `token_sink` mechanism: this module
    keeps a module-level `last_usage` dict the agent layer reads after each call.
    """
    global last_usage
    model = model or DEFAULT_MODEL
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(f"{BASE}/chat/completions", headers=headers, json=body, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            last_usage = {"prompt": usage.get("prompt_tokens", 0),
                          "completion": usage.get("completion_tokens", 0)}
            return [ch["message"]["content"] for ch in data.get("choices", [])]
        except Exception as e:  # noqa: BLE001 - any transport/HTTP error -> retry
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


# module-level usage sink (agent layer reads it after each call)
last_usage: dict = {"prompt": 0, "completion": 0}


def parse_order(text: str) -> int | None:
    """First integer in the model response; None if unparseable (instruction failure)."""
    m = _INT_RE.search(text or "")
    return int(m.group()) if m else None


def list_models() -> list[str]:
    """Verify which models the endpoint actually serves (/v1/models = truth)."""
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    resp = requests.get(f"{BASE}/models", headers=headers, timeout=30)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]
