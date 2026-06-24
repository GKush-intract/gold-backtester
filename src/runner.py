from __future__ import annotations

from typing import Optional

import pandas as pd

from .data_loader import resample
from .engine import BacktestConfig, BacktestResult, run_backtest as _engine_run
from .strategy import Strategy


def run_backtest(config: BacktestConfig, strategy: Strategy,
                 data: pd.DataFrame, htf: Optional[dict] = None) -> BacktestResult:
    """Public entry point shared by CLI/UI. Builds the HTF views the strategy declares,
    then delegates to the engine."""
    if htf is None:
        htf = {tf: resample(data, tf) for tf in getattr(strategy, "htf_timeframes", [])}
    return _engine_run(config, strategy, data, htf=htf)
