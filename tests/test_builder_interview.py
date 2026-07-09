# tests/test_builder_interview.py
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_get_client_returns_none_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # point at a nonexistent .env so a developer's real repo-root .env can't leak in
    assert interview.get_client(env_file=tmp_path / "missing.env") is None


def test_get_client_reads_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-test-123\n")
    assert interview.get_client(env_file=env) is not None


def test_turn_returns_text_only_and_appends_assistant_turn():
    client = _mock_client([_text_block("What timeframe do you want?")])
    history = [{"role": "user", "content": "EFI strategy"}]
    text, spec = interview.run_interview_turn(client, history)
    assert "timeframe" in text
    assert spec is None
    assert history[-1] == {"role": "assistant", "content": "What timeframe do you want?"}


def test_turn_returns_spec_and_acks_tool_call():
    client = _mock_client([_text_block("Here is the spec."), _tool_block(VALID_SPEC)])
    history = [{"role": "user", "content": "done"}]
    text, spec = interview.run_interview_turn(client, history)
    assert spec["name"] == "EFI Pullback"
    assert history[-2]["role"] == "assistant"
    assert history[-2]["content"][1]["type"] == "tool_use"
    ack = history[-1]
    assert ack["role"] == "user"
    assert ack["content"][0]["type"] == "tool_result"
    assert ack["content"][0]["tool_use_id"] == "tu_1"
    assert "is_error" not in ack["content"][0]
    # parallel tool use disabled on the request
    assert client.messages.create.call_args.kwargs["tool_choice"] == {
        "type": "auto", "disable_parallel_tool_use": True}


def test_invalid_spec_rejected_with_error_tool_result():
    bad = {"name": "x"}  # missing required keys
    client = _mock_client([_tool_block(bad)])
    history = [{"role": "user", "content": "done"}]
    text, spec = interview.run_interview_turn(client, history)
    assert spec is None
    assert "missing" in text.lower()
    err = history[-1]["content"][0]
    assert err["type"] == "tool_result" and err["is_error"] is True


def test_empty_parameters_list_is_valid():
    spec_no_params = dict(VALID_SPEC, parameters=[])
    client = _mock_client([_tool_block(spec_no_params)])
    text, spec = interview.run_interview_turn(client, [{"role": "user", "content": "done"}])
    assert spec is not None
    assert spec["parameters"] == []


def test_truncated_response_flagged():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[_text_block("partial answer")], stop_reason="max_tokens")
    text, spec = interview.run_interview_turn(client, [{"role": "user", "content": "hi"}])
    assert "truncated" in text


def _rev_tool_block(plan):
    return SimpleNamespace(type="tool_use", name="finalize_revision", input=plan, id="tu_r1")


VALID_PLAN = {"summary": "Add an H4 EMA trend filter gating long entries.",
              "changes": ["Add param htf_ema_period (int, 21, 3, 200)",
                          "Only enter when H4 close > H4 EMA(htf_ema_period)"]}


def test_revision_turn_question_passthrough():
    client = _mock_client([_text_block("Which timeframe should the filter use?")])
    history = [{"role": "user", "content": "add a trend filter"}]
    text, plan = interview.run_revision_turn(client, history, code="CODE", spec={"name": "X"})
    assert plan is None
    assert "timeframe" in text
    assert history[-1] == {"role": "assistant", "content": "Which timeframe should the filter use?"}
    # current code + spec are embedded in the system prompt
    system = client.messages.create.call_args.kwargs["system"]
    assert "CODE" in system and '"name": "X"' in system


def test_revision_turn_returns_plan_and_acks():
    client = _mock_client([_text_block("Plan ready."), _rev_tool_block(VALID_PLAN)])
    history = [{"role": "user", "content": "gate entries on H4 trend, EMA 21"}]
    text, plan = interview.run_revision_turn(client, history, code="CODE")
    assert plan == VALID_PLAN
    ack = history[-1]["content"][0]
    assert ack["type"] == "tool_result" and "is_error" not in ack


def test_revision_turn_rejects_empty_changes():
    client = _mock_client([_rev_tool_block({"summary": "do stuff", "changes": []})])
    text, plan = interview.run_revision_turn(client, [{"role": "user", "content": "x"}], code="C")
    assert plan is None
    assert "missing" in text.lower()
    assert "changes" in text
