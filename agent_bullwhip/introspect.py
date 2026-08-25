"""Introspection parsing: extract (order, confidence, reasoning) from LLM output.

The introspective agent asks the model to answer in a strict 3-line format:
    ORDER: <int>
    CONFIDENCE: <0.0-1.0>
    REASONING: <one sentence>

This parser is deliberately tolerant: it scans for the three labeled fields in
any order, and falls back to the classic first-integer order parse when the
model ignores the format (so the KB/reasoning upgrade never breaks the
baseline measurement).
"""
from __future__ import annotations

import re

_ORDER_RE = re.compile(r"^\s*ORDER\s*:\s*(-?\d+)", re.IGNORECASE | re.MULTILINE)
_CONF_RE = re.compile(
    r"^\s*CONFIDENCE\s*:\s*(0?\.\d+|1\.0|1|0)", re.IGNORECASE | re.MULTILINE
)
_REASON_RE = re.compile(r"^\s*REASONING\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_INT_RE = re.compile(r"-?\d+")


def parse_introspection(text: str) -> dict | None:
    """Parse an introspective response into {order, confidence, reasoning}.

    Returns None when no ORDER field is found (caller falls back to classic parse).
    """
    m_order = _ORDER_RE.search(text or "")
    if not m_order:
        return None
    order = int(m_order.group(1))
    m_conf = _CONF_RE.search(text or "")
    conf = float(m_conf.group(1)) if m_conf else None
    m_reason = _REASON_RE.search(text or "")
    reason = m_reason.group(1).strip() if m_reason else ""
    return {"order": order, "confidence": conf, "reasoning": reason}


def classic_order(text: str) -> int | None:
    """The original parse: first integer anywhere in the response."""
    m = _INT_RE.search(text or "")
    return int(m.group()) if m else None
