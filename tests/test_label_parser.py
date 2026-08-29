"""Test the ORDER:-label-aware order parser (the GLM reasoning fix).

GLM-style reasoning models emit THINKING blocks before the answer; the old
regex-based parser could silently grab a number from the THINKING text instead
of the actual ORDER label. parse_order_prefer_label must always prefer the
explicit ORDER: label when present.
"""
from agent_bullwhip.client import parse_order_prefer_label, parse_order


def test_prefers_label_over_thinking():
    # number appears in THINKING first; label carries the true order
    text = "THINKING: backlog is 12, order 7 units\nORDER: 4\nCONFIDENCE: 0.8\nREASONING: cover backlog"
    assert parse_order_prefer_label(text) == 4
    # the old parser would have grabbed 12
    assert parse_order(text) == 12


def test_plain_order_label():
    assert parse_order_prefer_label("ORDER: 5\nCONFIDENCE: 0.6") == 5


def test_order_with_dash_and_colon_variants():
    assert parse_order_prefer_label("ORDER -3\nCONFIDENCE: 0.9") == -3
    assert parse_order_prefer_label("order: 8") == 8
    assert parse_order_prefer_label("Order = 2\n") == 2


def test_no_label_falls_back_to_first_number():
    # models that don't emit a label still parse (backward compat)
    assert parse_order_prefer_label("12\nCONFIDENCE: 0.5") == 12


def test_none_when_nothing_parseable():
    assert parse_order_prefer_label("no numbers here") is None
    assert parse_order_prefer_label("") is None
