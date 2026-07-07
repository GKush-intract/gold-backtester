# tests/test_builder_validate.py
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.builder import codegen, validate

GOOD = '''\
from __future__ import annotations
from ...strategy import Strategy

class ValidateOK(Strategy):
    name = "Validate OK"
    params = {"n": ("int", 3, 1, 50, "sma period")}
    def on_bar(self, ctx):
        if ctx.position is None and ctx.index % 50 == 10:
            price = ctx.bar["close"]
            ctx.enter("long", 0.01, stop_loss=price - 5, take_profit=price + 5)
'''

BROKEN = "from ...strategy import Strategy\nclass Broken(Strategy):\n    name = undefined_name\n"

LOOPS = '''\
from ...strategy import Strategy
class Loops(Strategy):
    name = "Loops"
    def on_bar(self, ctx):
        while True:
            pass
'''

NO_TRADES = '''\
from ...strategy import Strategy
class NoTrades(Strategy):
    name = "No Trades"
    def on_bar(self, ctx):
        pass
'''


@pytest.fixture(scope="module")
def smoke_csv(tmp_path_factory):
    idx = pd.date_range("2026-01-01", periods=500, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    base = 2000 + rng.normal(0, 1.0, len(idx)).cumsum()
    df = pd.DataFrame({"timestamp": idx, "open": base, "high": base + 1,
                       "low": base - 1, "close": base + 0.2, "volume": 1.0})
    p = tmp_path_factory.mktemp("data") / "smoke.csv"
    df.to_csv(p, index=False)
    return p


def _write(code, stem):
    path = codegen.GENERATED_DIR / f"{stem}.py"
    path.write_text(code)
    return path


def test_good_strategy_passes(smoke_csv):
    path = _write(GOOD, "gen_validate_ok_test")
    try:
        result = validate.validate_strategy(path, smoke_csv, bars=500)
        assert result["ok"], result
        assert result["num_trades"] >= 1
    finally:
        path.unlink()


def test_broken_strategy_fails(smoke_csv):
    path = _write(BROKEN, "gen_validate_broken_test")
    try:
        result = validate.validate_strategy(path, smoke_csv, bars=500)
        assert not result["ok"]
        assert "undefined_name" in result["error"]
    finally:
        path.unlink()


def test_infinite_loop_times_out(smoke_csv):
    path = _write(LOOPS, "gen_validate_loops_test")
    try:
        result = validate.validate_strategy(path, smoke_csv, bars=500, timeout=4)
        assert not result["ok"]
        assert "infinite loop" in result["error"]
    finally:
        path.unlink()


def test_zero_trades_is_warning(smoke_csv):
    path = _write(NO_TRADES, "gen_validate_notrades_test")
    try:
        result = validate.validate_strategy(path, smoke_csv, bars=500)
        assert result["ok"]
        assert result.get("warning")
    finally:
        path.unlink()


def test_generate_validated_repairs_then_succeeds(smoke_csv):
    client = MagicMock()
    responses = [BROKEN, GOOD]  # first attempt broken, repair fixes it

    def fake_create(**kwargs):
        code = responses.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"```python\n{code}\n```")],
            stop_reason="end_turn")

    client.messages.create.side_effect = fake_create
    spec = {"name": "Repair Test", "parameters": []}
    path, result, code = validate.generate_validated(client, spec, csv_path=smoke_csv,
                                                     bars=500, max_attempts=3)
    try:
        assert result["ok"]
        assert "ValidateOK" in code
        assert client.messages.create.call_count == 2
    finally:
        path.unlink()


def test_generate_validated_all_attempts_fail(smoke_csv):
    client = MagicMock()

    def fake_create(**kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"```python\n{BROKEN}\n```")],
            stop_reason="end_turn")

    client.messages.create.side_effect = fake_create
    path, result, code = validate.generate_validated(
        client, {"name": "Always Broken", "parameters": []},
        csv_path=smoke_csv, bars=500, max_attempts=2)
    try:
        assert not result["ok"]
        # initial generate + 1 repair (no repair after the final failed validation)
        assert client.messages.create.call_count == 2
    finally:
        path.unlink()


def test_generate_validated_truncated_generation(smoke_csv):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="```python\npartial")],
        stop_reason="max_tokens")
    path, result, code = validate.generate_validated(
        client, {"name": "Truncated", "parameters": []}, csv_path=smoke_csv, bars=500)
    assert path is None
    assert not result["ok"]
    assert "truncated" in result["error"]
