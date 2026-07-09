import pandas as pd

from src.ui_replay import detect_indicator_defaults, parse_overlays, trade_bar_positions


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
