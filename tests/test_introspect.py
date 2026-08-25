"""Tests for the introspection parser (Wave 2)."""
from agent_bullwhip.introspect import parse_introspection, classic_order


def test_parse_full():
    out = parse_introspection(
        "THINKING: backlog high\nORDER: 12\nCONFIDENCE: 0.8\nREASONING: backlog is high, covered it\n"
    )
    assert out["order"] == 12
    assert out["confidence"] == 0.8
    assert out["reasoning"] == "backlog is high, covered it"
    assert out["thinking"] == "backlog high"


def test_parse_missing_confidence():
    out = parse_introspection("ORDER: 5\nREASONING: stable demand\n")
    assert out["order"] == 5
    assert out["confidence"] is None
    assert out["reasoning"] == "stable demand"


def test_parse_missing_order_returns_none():
    assert parse_introspection("CONFIDENCE: 0.9\nREASONING: nothing") is None


def test_parse_confidence_zero_one():
    assert parse_introspection("ORDER: 0\nCONFIDENCE: 1.0\nREASONING: x")["confidence"] == 1.0
    assert parse_introspection("ORDER: 0\nCONFIDENCE: 0\nREASONING: x")["confidence"] == 0.0


def test_parse_tolerant_order_position():
    out = parse_introspection("CONFIDENCE: 0.7\nORDER: 9\nREASONING: steady\n")
    assert out["order"] == 9


def test_classic_fallback():
    assert classic_order("I would order 14 units.") == 14
    assert classic_order("no numbers here") is None
