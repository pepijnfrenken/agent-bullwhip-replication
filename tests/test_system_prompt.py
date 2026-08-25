"""Tests for the system-placement prompt variant (Wave 2b)."""
from agent_bullwhip.agents import build_system_prompt, build_user_prompt, build_prompt

CTX = {
    "t": 5,
    "on_hand": 4,
    "backlog": 8,
    "outstanding": 2,
    "incoming_last": 4,
    "incoming_now": 8,
    "last_orders": [4, 4, 8, 12],
}


def test_system_prompt_contains_identity_and_kb():
    s = build_system_prompt("retailer", kb=True, introspect=True)
    assert "RETAILER" in s
    assert "DECISION PLAYBOOK" in s
    assert "kb-1" in s
    assert "CONFIDENCE" in s


def test_system_prompt_without_kb_has_no_playbook():
    s = build_system_prompt("retailer", kb=False)
    assert "DECISION PLAYBOOK" not in s
    assert "RETAILER" in s


def test_user_prompt_contains_state_not_identity():
    u = build_user_prompt("retailer", CTX, introspect=True)
    assert "Current week: 5" in u
    assert "on_hand" in u.lower() or "On-hand" in u
    assert "RETAILER" not in u.upper().replace("RETAILER", "")  # identity not duplicated
    assert "CONFIDENCE" in u or "ORDER" in u


def test_inline_prompt_still_works():
    p = build_prompt("retailer", CTX, introspect=True, kb=True)
    assert "DECISION PLAYBOOK" in p
    assert "Current week: 5" in p
    assert "CONFIDENCE" in p
