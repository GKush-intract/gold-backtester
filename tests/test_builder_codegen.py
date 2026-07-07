# tests/test_builder_codegen.py
import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.builder import codegen

GOOD_CODE = '''\
from __future__ import annotations

from ...strategy import Strategy


class TempGen(Strategy):
    name = "Temp Gen"
    params = {"n": ("int", 5, 1, 100, "period")}

    def on_bar(self, ctx):
        pass
'''


def test_extract_code_single_block():
    text = f"Here is your strategy:\n```python\n{GOOD_CODE}```\nEnjoy!"
    assert codegen.extract_code(text) == GOOD_CODE


def test_extract_code_takes_last_block():
    text = f"```python\n# draft\n```\nFinal:\n```python\n{GOOD_CODE}```"
    assert codegen.extract_code(text) == GOOD_CODE


def test_extract_code_bare():
    assert codegen.extract_code(GOOD_CODE) == GOOD_CODE


def test_extract_code_no_code_raises():
    with pytest.raises(ValueError):
        codegen.extract_code("I could not generate the strategy.")


def test_strategy_filename():
    now = dt.datetime(2026, 7, 7, 10, 42)
    assert codegen.strategy_filename("EFI Pullback!", now) == "gen_efi_pullback_20260707_1042.py"


def test_write_and_load():
    # writes into the real generated/ dir so package-relative import works; cleaned up below
    path = codegen.write_strategy_file(GOOD_CODE, "Temp Gen")
    try:
        assert path.parent == codegen.GENERATED_DIR
        cls = codegen.load_strategy_class(path)
        assert cls.name == "Temp Gen"
        assert cls.params["n"][1] == 5
    finally:
        path.unlink()


def test_generate_strategy_embeds_contract():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=f"```python\n{GOOD_CODE}```")])
    spec = {"name": "Temp Gen", "parameters": []}
    code = codegen.generate_strategy(client, spec)
    assert "class TempGen" in code
    system = client.messages.create.call_args.kwargs["system"]
    assert "class Strategy" in system          # strategy.py source embedded
    assert "MACrossover" in system             # example embedded


def _client_returning(text):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")
    return client


def test_revise_strategy_sends_code_and_request():
    client = _client_returning(f"```python\n{GOOD_CODE}```")
    codegen.revise_strategy(client, "OLD_CODE_MARKER", "use EMA 21")
    user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "OLD_CODE_MARKER" in user_msg
    assert "use EMA 21" in user_msg


def test_repair_strategy_sends_code_and_error():
    client = _client_returning(f"```python\n{GOOD_CODE}```")
    codegen.repair_strategy(client, "OLD_CODE_MARKER", "Traceback: NameError")
    user_msg = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "OLD_CODE_MARKER" in user_msg
    assert "NameError" in user_msg


def test_truncated_response_raises():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="```python\npartial")], stop_reason="max_tokens")
    with pytest.raises(ValueError, match="truncated"):
        codegen.generate_strategy(client, {"name": "x", "parameters": []})


def test_spec_sidecar_roundtrip(tmp_path):
    path = tmp_path / "gen_x.py"
    codegen.save_spec(path, {"name": "X", "parameters": []})
    assert codegen.spec_sidecar_path(path) == tmp_path / "gen_x.spec.json"
    assert codegen.load_spec(path) == {"name": "X", "parameters": []}


def test_load_spec_missing_or_corrupt(tmp_path):
    assert codegen.load_spec(tmp_path / "gen_missing.py") is None
    bad = tmp_path / "gen_bad.py"
    codegen.spec_sidecar_path(bad).write_text("not json{")
    assert codegen.load_spec(bad) is None
