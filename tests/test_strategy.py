import pandas as pd
from src.strategy import Context, Strategy, Order


def _data():
    idx = pd.date_range("2024-01-02", periods=3, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": [10, 11, 12], "high": [12, 13, 14], "low": [9, 10, 11],
         "close": [11, 12, 13], "volume": [1, 1, 1]}, index=idx)


def test_enter_records_order():
    ctx = Context(_data(), index=1, position=None, equity=1000, balance=1000, htf={})
    ctx.enter("long", size=2, stop_loss=10.0, take_profit=15.0, tag="t")
    assert isinstance(ctx._order, Order)
    assert ctx._order.direction == "long" and ctx._order.size == 2


def test_bar_and_history():
    ctx = Context(_data(), index=1, position=None, equity=1000, balance=1000, htf={})
    assert ctx.bar["close"] == 12
    assert len(ctx.history) == 2  # bars 0..1 inclusive


def test_size_for_risk():
    ctx = Context(_data(), index=0, position=None, equity=10000, balance=10000, htf={})
    # lots = (10000*0.01) / (|2000-1990| * 100 oz/lot) = 100 / 1000 = 0.1 lots
    assert ctx.size_for_risk(0.01, entry_price=2000, stop_price=1990) == 0.1
    assert ctx.size_for_risk(0.01, entry_price=2000, stop_price=2000) == 0.0


def test_strategy_param_merge():
    class S(Strategy):
        params = {"fast": ("int", 20, 1, 100, "fast ma")}
    s = S(fast=5)
    assert s.p["fast"] == 5
    assert S().p["fast"] == 20
