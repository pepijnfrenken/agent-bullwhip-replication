"""Tests for the verbal-consistency gate (Wave 2e ablation)."""
import json

from agent_bullwhip.agents import LLMAgent, LLMAgentConfig
from agent_bullwhip.introspect import parse_introspection


def _make_agent(**kw):
    base = dict(
        introspect=True,
        kb=True,
        kb_placement="pointer",
        conf_threshold=0.5,
        anchor_margin=6,
    )
    base.update(kw)
    cfg = LLMAgentConfig(**base)
    return LLMAgent("retailer", cfg)


def _sample(text: str):
    """Wrap a response body into the introspect format string."""
    return f"THINKING: x\nORDER: 0\nCONFIDENCE: 0.8\nREASONING: {text}"


def test_verbal_gate_fires_on_backlog_mention():
    """Thinking says 'cover the backlog' but order=0 with backlog>0 -> FULL floor override.

    State: backlog 40, outstanding 12, demand 8. The full order-up-to floor for this
    state is 3*smoothed(8) + 40 - (0 + 12 - 40) = 64+40-(-28)=92-ish — far above the
    anchor±6 clamp. The gate must produce a large positive order, not a tiny one.
    """
    agent = _make_agent(consistency_gate=True)
    ctx = {"t": 5, "on_hand": 0, "backlog": 40, "outstanding": 12,
           "incoming_last": 8, "incoming_now": 8, "last_orders": [8, 8, 8, 8, 8]}
    order = agent._decide_introspect(ctx, [_sample("I must cover the backlog first")])
    assert order is not None and order > 0, f"verbal gate should override 0, got {order}"
    tr = agent.traces[-1]
    assert tr["gated"] is True
    assert tr["gated_verbal"] is True
    assert tr["order"] == 0
    assert tr["order_used"] > 0
    # full floor: forecast starts 0 -> 0.5*8+0.5*0 = 4; target = 3*4 + 40 = 52;
    # IP = 0 + 12 - 40 = -28; floor = 52 - (-28) = 80? NO: OrderUpToAgent uses
    # theta*forecast - ip = 3*4 - (-28) = 12 + 28 = 40. So the correct full floor = 40.
    assert tr["order_used"] == 40, f"expected full floor 40, got {tr['order_used']}"


def test_verbal_gate_fires_on_order_up_to():
    agent = _make_agent(consistency_gate=True)
    ctx = {"t": 5, "on_hand": 0, "backlog": 30, "outstanding": 10,
           "incoming_last": 8, "incoming_now": 8, "last_orders": [8, 8, 8, 8, 8]}
    order = agent._decide_introspect(ctx, [_sample("use order-up-to with pipeline adjustment")])
    assert order is not None and order > 0
    assert agent.traces[-1]["gated_verbal"] is True


def test_verbal_gate_not_fired_when_disabled():
    agent = _make_agent(consistency_gate=False)
    ctx = {"t": 5, "on_hand": 0, "backlog": 40, "outstanding": 60,
           "incoming_last": 8, "incoming_now": 8, "last_orders": [8, 8, 8, 8, 8]}
    order = agent._decide_introspect(ctx, [_sample("I must cover the backlog first")])
    # gate disabled -> model order (0) passes through (plus wrappers)
    assert order == 0
    assert agent.traces[-1]["gated"] is False
    assert agent.traces[-1]["gated_verbal"] is False


def test_verbal_gate_not_fired_without_backlog():
    agent = _make_agent()
    ctx = {"t": 5, "on_hand": 10, "backlog": 0, "outstanding": 20,
           "incoming_last": 8, "incoming_now": 8, "last_orders": [8, 8, 8, 8, 8]}
    order = agent._decide_introspect(ctx, [_sample("cover the backlog, stable")])
    assert order == 0
    assert agent.traces[-1]["gated"] is False


def test_verbal_gate_not_fired_on_nonzero_order():
    agent = _make_agent(anchor_margin=None)  # no wrapper clamp — pure pass-through
    ctx = {"t": 5, "on_hand": 0, "backlog": 40, "outstanding": 60,
           "incoming_last": 8, "incoming_now": 8, "last_orders": [8, 8, 8, 8, 8]}
    resp = "ORDER: 12\nCONFIDENCE: 0.8\nREASONING: cover the backlog"
    order = agent._decide_introspect(ctx, [resp])
    assert order == 12
    assert agent.traces[-1]["gated"] is False


def test_parse_consistency_roundtrip():
    """The gate text (thinking+reasoning) parses the same way the runner does."""
    p = parse_introspection("THINKING: x\nORDER: 0\nCONFIDENCE: 0.8\nREASONING: must cover backlog\n")
    assert p["order"] == 0
    assert p["confidence"] == 0.8
