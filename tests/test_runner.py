import numpy as np
import pandas as pd

from src import runner
from src.strategies import get_strategy_registry
from src.engine import BacktestConfig


def _trending_data(n=300):
    idx = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    base = np.concatenate([np.linspace(2000, 2100, n // 2), np.linspace(2100, 1980, n - n // 2)])
    df = pd.DataFrame({
        "open": base, "high": base + 1.0, "low": base - 1.0, "close": base, "volume": 1.0,
    }, index=idx)
    df.index.name = "timestamp"
    return df


def test_registry_discovers_ma_crossover():
    reg = get_strategy_registry()
    assert any("crossover" in name.lower() for name in reg)


def test_runner_runs_ma_crossover():
    reg = get_strategy_registry()
    cls = next(c for name, c in reg.items() if "crossover" in name.lower())
    strat = cls(fast=5, slow=20, risk_pct=0.01)
    res = runner.run_backtest(BacktestConfig(), strat, _trending_data())
    assert res.equity_curve.shape[0] == 300
    assert res.trades.shape[0] >= 1
