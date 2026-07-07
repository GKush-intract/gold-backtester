# tests/test_builder_interview.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.builder import interview


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(spec):
    return SimpleNamespace(type="tool_use", name="finalize_spec", input=spec, id="tu_1")


VALID_SPEC = {
    "name": "EFI Pullback",
    "description": "Buy pullbacks when EMA-smoothed EFI crosses up",
    "timeframe": "m5",
    "buy_condition": "EFI(13) EMA crosses above 0",
    "sell_condition": "EFI crosses below 0 or stop hit",
    "tp_sl_strategy": "SL 2*ATR(14), TP 4*ATR(14)",
    "risk_management": "1% equity risk per trade",
    "parameters": [
        {"name": "efi_period", "type": "int", "default": 13, "min": 2, "max": 100,
         "help": "EFI EMA period"},
    ],
}


def _mock_client(content):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=content, stop_reason="end_turn")
    return client


def test_get_client_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert interview.get_client() is None


def test_turn_returns_text_only():
    client = _mock_client([_text_block("What timeframe do you want?")])
    text, spec, _ = interview.run_interview_turn(client, [{"role": "user", "content": "EFI strategy"}])
    assert "timeframe" in text
    assert spec is None


def test_turn_returns_spec_on_tool_call():
    client = _mock_client([_text_block("Here is the spec."), _tool_block(VALID_SPEC)])
    text, spec, raw = interview.run_interview_turn(client, [{"role": "user", "content": "done"}])
    assert spec["name"] == "EFI Pullback"
    assert raw[1].type == "tool_use"  # raw content returned for ack_spec()
    msgs = client.messages.create.call_args.kwargs["messages"]
    assert msgs[-1]["role"] == "user"  # history passed through unchanged


def test_invalid_spec_rejected():
    bad = {"name": "x"}  # missing required keys
    client = _mock_client([_tool_block(bad)])
    text, spec, _ = interview.run_interview_turn(client, [{"role": "user", "content": "done"}])
    assert spec is None
    assert "missing" in text.lower()


def test_ack_spec_appends_tool_result():
    history = [{"role": "user", "content": "done"}]
    content = [_text_block("spec ready"), _tool_block(VALID_SPEC)]
    interview.ack_spec(history, content)
    assert history[-2]["role"] == "assistant"
    assert history[-1]["role"] == "user"
    assert history[-1]["content"][0]["type"] == "tool_result"
    assert history[-1]["content"][0]["tool_use_id"] == "tu_1"
