import pandas as pd

from src.ui_replay import (build_chart_payload, compute_window, detect_indicator_defaults,
                           parse_overlays, trade_bar_positions)


def _index(n=10):
    return pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")


def test_trade_bar_positions_maps_times():
    idx = _index()
    trades = pd.DataFrame({
        "entry_time": [idx[2], idx[5]],
        "exit_time": [idx[4], idx[9]],
        "pnl": [10.0, -5.0],
    })
    t = trade_bar_positions(trades, idx)
    assert list(t["entry_idx"]) == [2, 5]
    assert list(t["exit_idx"]) == [4, 9]


def test_trade_bar_positions_clips_out_of_range():
    idx = _index()
    trades = pd.DataFrame({
        "entry_time": [idx[0] - pd.Timedelta("1h")],
        "exit_time": [idx[-1] + pd.Timedelta("1h")],
        "pnl": [0.0],
    })
    t = trade_bar_positions(trades, idx)
    assert t["entry_idx"].iloc[0] == 0
    assert t["exit_idx"].iloc[0] == len(idx) - 1


def test_trade_bar_positions_empty():
    t = trade_bar_positions(pd.DataFrame(columns=["entry_time", "exit_time"]), _index())
    assert len(t) == 0
    assert "entry_idx" in t.columns and "exit_idx" in t.columns


def test_detect_indicator_defaults_finds_base_emas_only():
    params = {"ema_period": 33, "htf_ema_period": 21, "atr_period": 14,
              "risk_pct": 1.0, "fast_ema_period": 5}
    assert detect_indicator_defaults(params) == "ema:5, ema:33"
    assert detect_indicator_defaults(None) == ""
    assert detect_indicator_defaults({"rsi_period": 14}) == ""


def test_parse_overlays():
    assert parse_overlays("ema:33, sma:50") == [("ema", 33), ("sma", 50)]
    assert parse_overlays("") == []
    assert parse_overlays("macd:12, ema:abc, ema:33, ema:33") == [("ema", 33)]


def test_compute_window_centers():
    lo, hi = compute_window(center=500, span=200, n=10_000)
    assert (lo, hi) == (400, 600)


def test_compute_window_widens_for_long_trade():
    # a 300-bar trade must fit fully with 20 bars of pad each side
    lo, hi = compute_window(center=500, span=200, n=10_000, trade_len=300)
    assert hi - lo >= 300 + 2 * 20
    assert lo <= 500 - 150 and hi >= 500 + 150


def test_compute_window_clamps_at_edges():
    lo, hi = compute_window(center=5, span=200, n=10_000)
    assert lo == 0 and hi == 200
    lo, hi = compute_window(center=9_998, span=200, n=10_000)
    assert hi == 9_999 and lo == 9_999 - 200
    lo, hi = compute_window(center=10, span=200, n=50)  # window bigger than data
    assert (lo, hi) == (0, 49)


def _mini_backtest():
    import numpy as np
    from src.engine import BacktestConfig, run_backtest
    from src.strategy import Strategy

    idx = pd.date_range("2026-01-05", periods=300, freq="5min", tz="UTC")
    rng = np.random.default_rng(3)
    base = 2000 + rng.normal(0, 1.0, 300).cumsum()
    data = pd.DataFrame({"open": base, "high": base + 1.5, "low": base - 1.5,
                         "close": base + 0.3, "volume": 1.0}, index=idx)
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    data.index.name = "timestamp"

    class T(Strategy):
        name = "t"

        def on_bar(self, ctx):
            if ctx.position is None and ctx.index % 40 == 10:
                price = ctx.bar["close"]
                ctx.enter("long", 0.1, stop_loss=price - 3, take_profit=price + 3)

    res = run_backtest(BacktestConfig(spread=0.0), T(), data)
    assert len(res.trades) >= 2
    return res, data


def test_chart_payload_structure():
    res, data = _mini_backtest()
    p = build_chart_payload(res, data, [("ema", 20)], sel=0, span=100)
    assert len(p["candles"]) == len(data)
    t = [c["time"] for c in p["candles"]]
    assert t == sorted(t) and all(isinstance(x, int) for x in t)
    assert t[1] - t[0] == 300  # 5min bars -> epoch seconds
    assert len(p["markers"]) == 2 * len(res.trades)
    assert p["sl"] and p["tp"] and p["conn"]      # selected trade segments present
    assert p["overlays"][0]["name"] == "EMA(20)"
    assert p["equity"]
    assert p["logical"]["from"] < p["logical"]["to"]


def test_chart_payload_no_selection():
    res, data = _mini_backtest()
    p = build_chart_payload(res, data, [], sel=None)
    assert p["sl"] == [] and p["tp"] == [] and p["conn"] == []
    assert p["logical"] is None
    assert p["window"] == [0, len(data) - 1]


def test_chart_payload_caps_large_datasets():
    res, data = _mini_backtest()
    p = build_chart_payload(res, data, [], sel=0, max_bars=100)
    lo, hi = p["window"]
    assert hi - lo <= 100
    assert len(p["candles"]) == hi - lo + 1
    # selected trade's logical window is inside the embedded slice
    if p["logical"] is not None:
        assert -1 <= p["logical"]["from"] < p["logical"]["to"] <= len(p["candles"])
