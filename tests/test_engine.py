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

    def __init__(self, size=1.0, sl=None, tp=None, trail=None, **kw):
        super().__init__(**kw)
        self.size, self.sl, self.tp, self.trail, self.done = size, sl, tp, trail, False

    def on_bar(self, ctx):
        if not self.done and ctx.position is None:
            ctx.enter("long", self.size, stop_loss=self.sl, take_profit=self.tp, trail=self.trail)
            self.done = True


class EnterOnceShort(EnterOnceLong):
    name = "enter_once_short"

    def on_bar(self, ctx):
        if not self.done and ctx.position is None:
            ctx.enter("short", self.size, stop_loss=self.sl, take_profit=self.tp, trail=self.trail)
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
    # size 2 lots = 200 oz: pnl = (107.9 - 100.1) * 200 - 1 commission = 1560 - 1 = 1559
    assert t["pnl"] == pytest.approx(1559.0)
    assert t["exit_reason"] == "tp"


def test_short_loser_pnl():
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 112, 100, 110)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0, commission_per_trade=0.0)
    strat = EnterOnceShort(size=1.0, sl=110, tp=80)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    # size 1 lot = 100 oz, short: pnl = (110 - 100) * 100 * -1 = -1000
    assert t["pnl"] == pytest.approx(-1000.0)
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
    # 1 lot = 100 oz: unrealized = (105 - 100) * 100 = 500
    assert ec["equity"].iloc[-1] == pytest.approx(cfg.opening_balance + 500)


def test_equity_zero_stops_run():
    # Huge long with no stop; price collapses so mark-to-market equity goes <= 0 -> run halts.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100, 100, 80), (80, 80, 80, 80)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1000.0, sl=None, tp=None)
    res = run_backtest(cfg, strat, data)
    assert res.stopped_out is True
    assert len(res.equity_curve) < len(data)  # halted before the final bar


def test_entry_bar_bracket_not_checked():
    # Convention: a position fills at a bar's open; its SL/TP are first checked on the NEXT bar.
    # Entry fills at bar1 open=100 with stop=95. Bar1's own low (90) must NOT stop it out;
    # the stop fires only on bar2 (low 90 <= 95). bars_held therefore == 1, not 0.
    data = make_data([(100, 100, 100, 100), (100, 100, 90, 100), (100, 100, 90, 100)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    strat = EnterOnceLong(size=1.0, sl=95, tp=1000)
    res = run_backtest(cfg, strat, data)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(95.0)
    assert t["bars_held"] == 1  # exited on the bar AFTER entry, not the entry bar


def test_trailing_stop_long_locks_profit():
    # Enter long @100 (bar1 open), trail=5 -> initial level 95. bar2 high 110 lifts level to 105.
    # bar3 retraces (low 104 <= 105) -> exit at the trailed level 105 with profit.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 110, 100, 108), (108, 108, 104, 104)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    res = run_backtest(cfg, EnterOnceLong(size=1.0, trail=5.0), data)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "trailing_stop"
    assert t["exit_price"] == pytest.approx(105.0)
    assert t["pnl"] == pytest.approx(500.0)  # (105-100)*1 lot*100 oz


def test_trailing_stop_long_initial_acts_as_stop():
    # Price never advances; trail's initial level (entry-5=95) acts as the stop.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 93, 96)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    res = run_backtest(cfg, EnterOnceLong(size=1.0, trail=5.0), data)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "trailing_stop"
    assert t["exit_price"] == pytest.approx(95.0)
    assert t["pnl"] == pytest.approx(-500.0)


def test_trailing_stop_gap_through_fills_at_open():
    # bar2 gaps below the initial trail level (95) at the open -> fill at the worse open price.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100), (90, 92, 88, 91)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    res = run_backtest(cfg, EnterOnceLong(size=1.0, trail=5.0), data)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "trailing_stop"
    assert t["exit_price"] == pytest.approx(90.0)


def test_trailing_stop_short_locks_profit():
    # Short @100 (bar1 open), trail=5 -> initial level 105. bar2 low 90 lowers level to 95.
    # bar3 rallies (high 96 >= 95) -> exit at 95 with profit.
    data = make_data([(100, 100, 100, 100), (100, 100, 100, 100),
                      (100, 100, 90, 92), (92, 96, 92, 95)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    res = run_backtest(cfg, EnterOnceShort(size=1.0, trail=5.0), data)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "trailing_stop"
    assert t["exit_price"] == pytest.approx(95.0)
    assert t["pnl"] == pytest.approx(500.0)  # short: (100-95)*1 lot*100 oz


class PartialCloser(Strategy):
    """Enter long on bar 0 (fills bar 1 open), request a fractional close on bar 2."""
    name = "partial_closer"

    def __init__(self, fraction, size=1.0, sl=90.0, **kw):
        super().__init__(**kw)
        self.fraction, self.size, self.sl, self.entered = fraction, size, sl, False

    def on_bar(self, ctx):
        if not self.entered and ctx.position is None:
            ctx.enter("long", self.size, stop_loss=self.sl)
            self.entered = True
        elif ctx.position is not None and ctx.index == 2:
            ctx.close(reason="scale_out", fraction=self.fraction)


def test_partial_close_books_fraction_and_keeps_rest():
    data = make_data([(100, 100, 100, 100)] * 3 + [(110, 110, 110, 110), (120, 120, 120, 120)])
    cfg = BacktestConfig(spread=0.0, slippage=0.0)
    res = run_backtest(cfg, PartialCloser(fraction=0.5), data)
    assert len(res.trades) == 1                       # remainder still open at the end
    t = res.trades.iloc[0]
    assert t["size"] == pytest.approx(0.5)
    assert t["exit_reason"] == "scale_out"
    assert t["exit_price"] == pytest.approx(110.0)    # fills at the NEXT bar open
    assert t["pnl"] == pytest.approx((110 - 100) * 0.5 * 100)
    ec = res.equity_curve
    assert ec["balance"].iloc[-1] == pytest.approx(cfg.opening_balance + 500)
    # remaining half marked to market at 120
    assert ec["equity"].iloc[-1] == pytest.approx(cfg.opening_balance + 500 + 20 * 0.5 * 100)


def test_partial_close_prorates_initial_risk():
    # entry 100 (sl 90 -> total risk $1000); half out at 110 (+1R), stop the rest at 90 (-1R)
    data = make_data([(100, 100, 100, 100)] * 3 + [(110, 110, 110, 110)] * 2
                     + [(90, 90, 89, 89)])
    res = run_backtest(BacktestConfig(spread=0.0, slippage=0.0), PartialCloser(fraction=0.5), data)
    assert len(res.trades) == 2
    scale, stop = res.trades.iloc[0], res.trades.iloc[1]
    assert scale["initial_risk"] == pytest.approx(500.0)
    assert scale["r_multiple"] == pytest.approx(1.0)
    assert stop["size"] == pytest.approx(0.5)
    assert stop["exit_reason"] == "stop"
    assert stop["initial_risk"] == pytest.approx(500.0)
    assert stop["r_multiple"] == pytest.approx(-1.0)


def test_close_fraction_one_is_full_close():
    data = make_data([(100, 100, 100, 100)] * 3 + [(110, 110, 110, 110)])
    res = run_backtest(BacktestConfig(spread=0.0), PartialCloser(fraction=1.0), data)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["size"] == pytest.approx(1.0)


def test_close_invalid_fraction_ignored():
    data = make_data([(100, 100, 100, 100)] * 4)
    res = run_backtest(BacktestConfig(spread=0.0), PartialCloser(fraction=0.0), data)
    assert len(res.trades) == 0  # request ignored; position stays open


def test_partial_fraction_rounding_to_zero_lots_ignored():
    # 0.01 lots * 0.5 rounds down to 0 lots -> ignored, position intact
    data = make_data([(100, 100, 100, 100)] * 4)
    res = run_backtest(BacktestConfig(spread=0.0), PartialCloser(fraction=0.5, size=0.01), data)
    assert len(res.trades) == 0
