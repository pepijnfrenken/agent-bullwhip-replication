"""Tests for pointer-mode KB and thinking-trace capture (Wave 2c)."""
from agent_bullwhip.agents import build_prompt
from agent_bullwhip.introspect import parse_introspection

CTX = {
    "t": 5, "on_hand": 4, "backlog": 8, "outstanding": 2,
    "incoming_last": 4, "incoming_now": 8, "last_orders": [4, 4, 8, 12],
}


def test_pointer_prompt_has_no_kb_contents():
    p = build_prompt("retailer", CTX, kb=True, kb_pointer=True, introspect=True)
    assert "END PLAYBOOK POINTER" in p
    assert "kb-1" not in p          # rule IDs are contents — must NOT leak
    assert "order at most ~1.5-2x" not in p   # specific rule text must NOT leak
    assert "self-check" in p
    # the introspect instruction should not force-cite a specific rule id
    assert "which playbook rule applies" not in p


def test_inline_prompt_has_kb_contents():
    p = build_prompt("retailer", CTX, kb=True, kb_pointer=False, introspect=True)
    assert "DECISION PLAYBOOK" in p
    assert "kb-1" in p              # contents ARE present inline


def test_pointer_prompt_asks_self_check():
    p = build_prompt("retailer", CTX, kb=True, kb_pointer=True)
    assert "own knowledge" in p
    assert "self-check" in p


def test_introspection_parses_thinking():
    out = parse_introspection(
        "THINKING: demand step, apply kb-1\n"
        "ORDER: 12\nCONFIDENCE: 0.8\nREASONING: backlog covered\n"
    )
    assert out["thinking"] == "demand step, apply kb-1"
    assert out["order"] == 12
    assert out["confidence"] == 0.8


def test_introspection_thinking_optional():
    out = parse_introspection("ORDER: 5\nCONFIDENCE: 0.6\nREASONING: steady\n")
    assert out["thinking"] == ""
