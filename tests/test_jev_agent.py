"""Offline tests for the Jev agent — the API is monkeypatched, so no key needed.

These lock the *arithmetic*: the model supplies a choice/probability/confidence,
the code composes the order. If someone later "simplifies" the composition, the
numbers here change and the audit trail breaks — that is what the tests prevent.
"""
from __future__ import annotations

import pytest

from agent_bullwhip import jev_agent
from agent_bullwhip.jev_agent import JevAgent, JevAgentConfig

CTX = {"t": 1, "on_hand": 0, "backlog": 0, "outstanding": 0, "incoming_last": 10,
       "last_orders": [10]}
# qhat = 0.5*10 + 0.5*0 = 5.0 ; ip = 0 ; anchor = 3.0*5.0 = 15


def fake_response(answers: dict, in_tok: int = 312, out_tok: int = 34):
    def _ask(state, questions, model=None, timeout=None, retries=3):
        assert isinstance(state, dict) and "state" in state, "state must be structured"
        return {"model": "jev-latest", "answers": answers,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok}}
    return _ask


def test_mult_argmax_composes_order(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "coverage": {"type": "choice", "choice": "2.0", "confidence": 0.9,
                     "probabilities": {"1.0": 0.1, "2.0": 0.9}},
        "below_target": {"type": "noul", "noul": 0.7},
        "backlog_pressure": {"type": "score", "score": 0.0, "legend": {}, "confidence": 0.9,
                             "probabilities": {"0": 1.0}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="choice_mult", warm_start=False))
    assert a.decide(CTX) == 10          # 2.0 * 5.0 - 0
    assert a.last_decision_meta["is_fallback"] is False
    assert a.last_decision_meta["confidence"] == 0.9
    assert a.prompt_tokens == 312 and a.completion_tokens == 34


def test_mult_expectation_uses_full_distribution(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "coverage": {"type": "choice", "choice": "1.0", "confidence": 0.5,
                     "probabilities": {"1.0": 0.4, "2.0": 0.6}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="expectation", warm_start=False))
    assert a.decide(CTX) == 8           # E[k] = 1.6 -> 1.6*5.0 = 8.0


def test_grid_mode(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "order": {"type": "choice", "choice": "10", "confidence": 0.8,
                  "probabilities": {"0": 0.2, "10": 0.8}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="choice_grid", warm_start=False))
    assert a.decide(CTX) == 10


def test_low_confidence_gate_falls_back_to_anchor(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "coverage": {"type": "choice", "choice": "2.0", "confidence": 0.2,
                     "probabilities": {"1.0": 0.4, "2.0": 0.6}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="choice_mult", conf_threshold=0.5, warm_start=False))
    assert a.decide(CTX) == 15          # anchor: 3.0 * 5.0 - 0
    assert a.last_decision_meta["gate_fired"] is True
    assert a.gate_fires == 1


def test_high_confidence_does_not_gate(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "coverage": {"type": "choice", "choice": "2.0", "confidence": 0.95,
                     "probabilities": {"2.0": 0.95}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="choice_mult", conf_threshold=0.5, warm_start=False))
    assert a.decide(CTX) == 10
    assert a.last_decision_meta["gate_fired"] is False


def test_anchor_margin_clamp(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "order": {"type": "choice", "choice": "40", "confidence": 1.0,
                  "probabilities": {"40": 1.0}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="choice_grid", anchor_margin=6, warm_start=False))
    assert a.decide(CTX) == 21          # anchor 15 + margin 6


def test_reads_mode_composes_in_code(monkeypatch):
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "below_target": {"type": "noul", "noul": 0.9},
        "backlog_pressure": {"type": "score", "score": 2.0, "legend": {}, "confidence": 0.8,
                             "probabilities": {"2": 1.0}},
        "demand_trend": {"type": "score", "score": 2.0, "legend": {}, "confidence": 0.8,
                         "probabilities": {"2": 1.0}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="reads", warm_start=False))
    order = a.decide(CTX)
    assert order >= 0
    assert a.last_decision_meta["detail"]["mult"] > 1.0


def test_api_failure_falls_back(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("HTTP 529 overloaded")
    monkeypatch.setattr(jev_agent, "ask", boom)
    a = JevAgent("retailer", JevAgentConfig(mode="choice_mult", warm_start=False))
    assert a.decide(CTX) == 15          # anchor fallback
    assert a.failures == 1
    assert a.last_decision_meta["is_fallback"] is True


def test_missing_key_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev_agent, "ENV_FILE", tmp_path / "nope.env")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        jev_agent.ask({"a": 1}, {"q": {"type": "noul", "instructions": "x?"}})


def test_warm_start_replays_order_history(monkeypatch):
    """mid-game probe states must not be treated as a cold start."""
    monkeypatch.setattr(jev_agent, "ask", fake_response({
        "coverage": {"type": "choice", "choice": "2.0", "confidence": 0.9,
                     "probabilities": {"2.0": 0.9}},
    }))
    a = JevAgent("retailer", JevAgentConfig(mode="choice_mult", warm_start=True))
    # history [10] -> q_hat = 5.0, then the week's update -> 7.5 ; 2.0*7.5 - 0 = 15
    assert a.decide(CTX) == 15
    assert a.anchor(CTX) == 22          # 3.0 * 7.5, forecast must NOT step again
