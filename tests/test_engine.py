import pandas as pd
import pytest

from src.engine import BacktestConfig, run_backtest
from src.strategy import Strategy


def make_data(bars):
    """bars: list of (open, high, low, close)."""
    idx = pd.date_range("2024-01-02", periods=len(bars), freq="5min", tz="UTC")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "timestamp"
    return df


class EnterOnceLong(Strategy):
    """Enter long on the first bar; fixed SL/TP; never re-enter."""
    name = "enter_once_long"

    def __init__(self, size=1.0, sl=None, tp=None, **kw):
        super().__init__(**kw)
        self.size, self.sl, self.tp, self.done = size, sl, tp, False

    def on_bar(self, ctx):
        if not self.done and ctx.position is None:
            ctx.enter("long", self.size, stop_loss=self.sl, take_profit=self.tp)
            self.done = True


class EnterOnceShort(EnterOnceLong):
    name = "enter_once_short"

    def on_bar(self, ctx):
        if not self.done and ctx.position is None:
            ctx.enter("short", self.size, stop_loss=self.sl, take_profit=self.tp)
            self.done = True


def test_no_lookahead_fills_next_open():
    # Signal on bar 0; must fill at bar 1 open, not bar 0 close.
    data = make_data([(100, 101, 99, 100), (110, 111, 109, 110), (110, 111, 109, 110)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=50, tp=1000)  # wide so it stays open
    res = run_backtest(cfg, strat, data)
    assert strat.done
    assert res.equity_curve["equity"].iloc[-1] == pytest.approx(cfg.opening_balance)


def test_long_winner_pnl_with_costs():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 100, 105)])
    cfg = BacktestConfig(spread=0.2, slippage=0.0, commission_per_trade=1.0)
    strat = EnterOnceLong(size=2.0, sl=90, tp=108)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    assert t["entry_price"] == pytest.approx(100.1)
    assert t["exit_price"] == pytest.approx(107.9)
    assert t["pnl"] == pytest.approx(14.6)
    assert t["exit_reason"] == "tp"


def test_short_loser_pnl():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 112, 100, 110)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0, commission_per_trade=0.0)
    strat = EnterOnceShort(size=1.0, sl=110, tp=80)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    assert t["pnl"] == pytest.approx(-10.0)
    assert t["exit_reason"] == "stop"


def test_intrabar_stop_first_vs_tp_first():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 94, 100)])
    strat_kw = dict(size=1.0, sl=95, tp=108)
    res_stop = run_backtest(BacktestConfig(spread=0, slippage=0), EnterOnceLong(**strat_kw), data)
    res_tp = run_backtest(BacktestConfig(spread=0, slippage=0, intrabar="tp_first"),
                          EnterOnceLong(**strat_kw), data)
    assert res_stop.trades.iloc[0]["exit_reason"] == "stop"
    assert res_tp.trades.iloc[0]["exit_reason"] == "tp"


def test_gap_through_stop_fills_at_open():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (90, 92, 88, 91)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=95, tp=200)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    assert t["exit_price"] == pytest.approx(90.0)
    assert t["exit_reason"] == "stop"


def test_equity_vs_balance_open_position():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 105, 100, 105)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=50, tp=1000)
    res = run_backtest(cfg, strat, data)
    ec = res.equity_curve
    assert ec["balance"].iloc[-1] == pytest.approx(cfg.opening_balance)
    assert ec["equity"].iloc[-1] == pytest.approx(cfg.opening_balance + 5)


def test_equity_zero_stops_run():
    # Huge long with no stop; price collapses so mark-to-market equity goes <= 0 -> run halts.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100, 100, 80), (80, 80, 80, 80)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1000.0, sl=None, tp=None)
    res = run_backtest(cfg, strat, data)
    assert res.stopped_out is True
    assert len(res.equity_curve) < len(data)  # halted before the final bar
