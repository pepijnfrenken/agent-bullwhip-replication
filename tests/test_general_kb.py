"""Tests for the generalizable (domain-agnostic) KB (Wave 2d)."""
from agent_bullwhip.agents import build_prompt, _kb_text

CTX = {
    "t": 5, "on_hand": 4, "backlog": 8, "outstanding": 2,
    "incoming_last": 4, "incoming_now": 8, "last_orders": [4, 4, 8, 12],
}


def test_general_kb_loaded():
    t = _kb_text("GENERAL_KB.md")
    assert "Principle 1" in t
    assert "domain-agnostic" in t


def test_general_kb_has_no_domain_constants():
    t = _kb_text("GENERAL_KB.md")
    # the general KB must NOT contain beer-game-specific constants/formulas
    assert "beer" not in t.lower() or "canonical example" in t  # only as example
    assert "2-week" not in t
    assert "backlog + this-week" not in t
    assert "alpha ~0.5" not in t


def test_build_prompt_with_general_kb():
    p = build_prompt("retailer", CTX, kb=True, kb_file="GENERAL_KB.md", introspect=True)
    assert "Principle 1" in p
    assert "kb-1" not in p          # domain rule IDs must NOT leak into general KB
    assert "2-week" not in p


def test_build_prompt_defaults_to_domain_kb():
    p = build_prompt("retailer", CTX, kb=True, introspect=True)
    assert "kb-1" in p              # default is the domain playbook
    assert "Principle 1" not in p
