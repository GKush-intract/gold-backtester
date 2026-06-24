import numpy as np
import pandas as pd

from src.engine import BacktestConfig, BacktestResult
from src import metrics


def _result(pnls, rs, opening=10_000):
    trades = pd.DataFrame({
        "pnl": pnls, "r_multiple": rs,
        "entry_time": pd.date_range("2024-01-02", periods=len(pnls), freq="1h", tz="UTC"),
        "exit_time": pd.date_range("2024-01-02 00:30", periods=len(pnls), freq="1h", tz="UTC"),
        "bars_held": [6] * len(pnls),
    })
    eq = opening + np.cumsum([0.0] + list(pnls))
    idx = pd.date_range("2024-01-02", periods=len(eq), freq="1h", tz="UTC")
    ec = pd.DataFrame({"equity": eq, "balance": eq}, index=idx)
    ec["peak"] = ec["equity"].cummax()
    ec["drawdown"] = ec["equity"] - ec["peak"]
    ec["drawdown_pct"] = ec["drawdown"] / ec["peak"]
    return BacktestResult(ec, trades, BacktestConfig(opening_balance=opening),
                          idx[0], idx[-1], 3600.0)


def test_basic_stats():
    m, df = metrics.compute_metrics(_result([100, -50, 200, -50], [2, -1, 4, -1]))
    assert m["num_trades"] == 4
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == (300) / (100)
    assert m["expectancy_r"] == np.mean([2, -1, 4, -1])
    assert isinstance(df, pd.DataFrame)


def test_no_trades_flag():
    m, _ = metrics.compute_metrics(_result([], []))
    assert m["no_trades"] is True
    assert m["num_trades"] == 0


def test_profit_factor_inf_when_no_losses():
    m, _ = metrics.compute_metrics(_result([100, 50], [1, 1]))
    assert m["profit_factor"] == float("inf")


def test_max_drawdown():
    m, _ = metrics.compute_metrics(_result([100, -50, 200, -50], [1, -1, 1, -1]))
    assert m["max_drawdown_abs"] <= 0
    assert m["max_drawdown_pct"] <= 0
