"""WAVE6-FIX answer contract (docs/waves/WAVE6-FIX_SPEC.md §3-§4): the paper's
`order_quantity` JSON request, its strict parser, and telemetry helpers.

Torch-free. The wave-6 default path (raw continuation, first-integer parse) is
untouched; this module is only used when `--prompt-format chat_contract`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .agents import build_prompt
from .qwen_chat import render_nothink

LEGACY_REPLY = ("Reply with ONLY an integer: the number of units to order this "
                "week (0 or more).")

# Paper App. A.2 task block with three disclosed edits (braces, "cases of beer"
# -> "units", ASCII quotes). First line = U+2014 x9 separator. Exact bytes.
CONTRACT_TAIL = (
    "\u2014" * 9 + "\n"
    "Your Task:\n"
    "Decide how many units to order from your supplier this week based on your current situation.\n"
    "\n"
    "Start your response with a JSON object **on its own line** in the following exact format:\n"
    '{"order_quantity": <number_of_units>}\n'
    "\n"
    "Important:\n"
    "- Replace '<number_of_units>' with your actual numeric decision.\n"
    "- Do not add any text, notes, or punctuation after the JSON.\n"
    "- This will be parsed by a program, so the format must be valid and exact.\n"
    "\n"
    "Example (your response should end like this):\n"
    '{"order_quantity": 5}'
)

CONTRACT_RE = re.compile(r'^\{\s*"order_quantity"\s*:\s*([0-9]{1,4})\s*\}$')
LOOSE_JSON_RE = re.compile(r'"order_quantity"\s*:\s*[0-9]+')


@dataclass(frozen=True)
class ParseResult:
    value: int | None
    exact: bool


def render_chat_nothink(user: str) -> str:
    return render_nothink([{"role": "user", "content": user}])


def build_contract_user(role: str, ctx: dict, prompt_variant: str = "default") -> str:
    """The user message: the repo's state text with only the answer request swapped."""
    p = build_prompt(role, ctx, prompt_variant)
    if not p.endswith(LEGACY_REPLY):
        raise AssertionError("build_prompt no longer ends with LEGACY_REPLY")
    return p[: -len(LEGACY_REPLY)] + CONTRACT_TAIL


def build_contract_prompt(role: str, ctx: dict, prompt_variant: str = "default") -> str:
    return render_chat_nothink(build_contract_user(role, ctx, prompt_variant))


def parse_contract(text: str | None) -> ParseResult:
    t = (text or "").strip()
    first = t.split("\n", 1)[0].strip()
    m = CONTRACT_RE.match(first)
    if not m:
        return ParseResult(value=None, exact=False)
    return ParseResult(value=int(m.group(1)), exact=(t == first))


def has_loose_json(text: str | None) -> bool:
    return LOOSE_JSON_RE.search(text or "") is not None


def count_digit_tokens(completion_ids: list, parsed: list,
                       decode_token: Callable[[int], str]) -> tuple[int, int]:
    """(digit tokens in parsed rows, all completion tokens), over rows with ids."""
    digits = total = 0
    for ids, ok in zip(completion_ids, parsed):
        if ids is None:
            continue
        total += len(ids)
        if not ok:
            continue
        for i in ids:
            s = decode_token(int(i)).strip()
            if s and s.isascii() and s.isdigit():
                digits += 1
    return digits, total
