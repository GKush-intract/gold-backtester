import numpy as np
import pandas as pd

from src.engine import BacktestConfig, run_backtest
from src.strategies.smc_sweep import SMCLiquiditySweep, _pivots, _wilder_atr


def make_data(bars):
    """bars: list of (open, high, low, close)."""
    idx = pd.date_range("2024-01-02", periods=len(bars), freq="5min", tz="UTC")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    df.index.name = "timestamp"
    return df


# Test params: tiny windows so a hand-built sequence reaches every stage quickly.
TEST_PARAMS = dict(
    pivot_n=1, pool_lookback=50, atr_len=3, disp_atr_mult=1.0, disp_body_ratio=0.5,
    fib_lo=0.5, fib_hi=0.618, setup_expiry=50, sl_buffer_atr=0.1, rr=2.0,
    risk_pct=0.01, be_trigger_r=1.0, use_structure_exit=True,
)

# A clean bullish sweep -> MSS -> displacement -> FVG-tap -> confirmation, then a run to TP.
#        open,   high,   low,    close
BULLISH = [
    (2000, 2001, 1999, 2000),   # 0
    (2000, 2006, 1999, 2005),   # 1  swing HIGH (ref to break) = 2006
    (2004, 2005, 2000, 2001),   # 2
    (2001, 2002, 1995, 1996),   # 3  swing LOW (liquidity pool) = 1995
    (1997, 2000, 1996, 1999),   # 4
    (1998, 1999, 1990, 1996),   # 5  SWEEP: low 1990 < 1995, close 1996 > 1995
    (1997, 2002, 1996, 2001),   # 6  displacement up
    (2003, 2010, 2002, 2009),   # 7  MSS: close 2009 > 2006; FVG = (high[5]=1999, low[7]=2002)
    (1999, 2002, 1999, 2001),   # 8  taps fib∩FVG zone (1999,2000), bullish confirm -> ENTER
    (2002, 2004, 2001, 2003),   # 9  entry fills here at open
    (2004, 2010, 2003, 2009),   # 10
    (2010, 2026, 2009, 2025),   # 11 high 2026 >= TP -> exit
    (2025, 2027, 2024, 2026),   # 12
]


def test_helpers_pivots_and_atr():
    df = make_data(BULLISH)
    ph_idx, ph_val = _pivots(df["high"].to_numpy(float), 1, "high")
    pl_idx, pl_val = _pivots(df["low"].to_numpy(float), 1, "low")
    assert 1 in ph_idx and ph_val[list(ph_idx).index(1)] == 2006   # the 2006 swing high
    assert 3 in pl_idx and pl_val[list(pl_idx).index(3)] == 1995   # the 1995 swing low
    atr = _wilder_atr(df["high"].to_numpy(float), df["low"].to_numpy(float),
                      df["close"].to_numpy(float), 3)
    assert np.isnan(atr[2]) and atr[3] > 0                          # warmup then defined


def test_bullish_setup_enters_long_with_structural_stop():
    df = make_data(BULLISH)
    cfg = BacktestConfig(opening_balance=10_000.0, spread=0.0, max_leverage=0.0)
    res = run_backtest(cfg, SMCLiquiditySweep(**TEST_PARAMS), df)

    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["direction"] == "long"
    assert tr["tag"] == "smc_long"
    assert tr["stop_loss"] < 1990                  # structural: below the swept low (1990)
    assert tr["take_profit"] > tr["entry_price"]
    assert tr["exit_reason"] == "tp"


def test_sweep_without_mss_does_not_trade():
    # Sweep the low, but price keeps drifting down and never breaks the prior swing high.
    bars = [
        (2000, 2001, 1999, 2000),   # 0
        (2000, 2006, 1999, 2005),   # 1  swing high 2006
        (2004, 2005, 2000, 2001),   # 2
        (2001, 2002, 1995, 1996),   # 3  swing low 1995
        (1997, 2000, 1996, 1999),   # 4
        (1998, 1999, 1990, 1996),   # 5  sweep
        (1995, 1997, 1992, 1993),   # 6  drifts down, no MSS
        (1993, 1995, 1989, 1990),   # 7
        (1990, 1992, 1986, 1988),   # 8
    ]
    res = run_backtest(BacktestConfig(spread=0.0), SMCLiquiditySweep(**TEST_PARAMS), make_data(bars))
    assert len(res.trades) == 0


def test_breakeven_stop_moves_after_one_r():
    # Enter long, run +1R to arm breakeven, then reverse down through entry. Exit should be at
    # ~breakeven (the moved stop), not the original structural stop far below.
    bars = BULLISH[:9] + [
        (2002, 2004, 2001, 2003),   # 9  entry fills at open ~2002
        (2004, 2016, 2003, 2015),   # 10 high 2016 -> arms breakeven (entry + 1R, R≈13)
        (2003, 2004, 1999, 2000),   # 11 trades back down through entry -> stop at breakeven
        (2000, 2001, 1995, 1996),   # 12
    ]
    cfg = BacktestConfig(opening_balance=10_000.0, spread=0.0, max_leverage=0.0,
                         intrabar="stop_first")
    res = run_backtest(cfg, SMCLiquiditySweep(**TEST_PARAMS), make_data(bars))
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    # exited near entry (breakeven), well above the original structural stop (~1989)
    assert tr["exit_price"] > 1995
    assert tr["exit_reason"] in ("stop", "trailing_stop")
