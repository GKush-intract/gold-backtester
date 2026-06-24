from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import Context, Position, Strategy


@dataclass
class BacktestConfig:
    opening_balance: float = 10_000.0
    spread: float = 0.30
    slippage: float = 0.0
    commission_per_trade: float = 0.0
    commission_per_unit: float = 0.0
    intrabar: str = "stop_first"   # "stop_first" | "tp_first" | "optimistic"
    entry_fill: str = "next_open"  # "next_open" only in v1 (kept for forward-compat)


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    size: float
    entry_price: float
    exit_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: float
    r_multiple: Optional[float]
    exit_reason: str
    tag: str
    bars_held: int
    initial_risk: Optional[float]


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame   # index=time, cols: equity, balance, peak, drawdown, drawdown_pct
    trades: pd.DataFrame
    config: BacktestConfig
    data_start: object
    data_end: object
    timeframe_seconds: float
    stopped_out: bool = False


# --- cost helpers: every fill pays half-spread + slippage adverse to its side ---
def _buy_fill(base, cfg):
    return base + cfg.spread / 2 + cfg.slippage


def _sell_fill(base, cfg):
    return base - cfg.spread / 2 - cfg.slippage


def _commission(size, cfg):
    return cfg.commission_per_trade + cfg.commission_per_unit * size


def _trade_pnl(pos: Position, exit_fill: float, cfg: BacktestConfig) -> float:
    sign = 1 if pos.direction == "long" else -1
    gross = (exit_fill - pos.entry_price) * pos.size * sign
    return gross - _commission(pos.size, cfg)


def _unrealized(pos: Optional[Position], price: float) -> float:
    if pos is None:
        return 0.0
    sign = 1 if pos.direction == "long" else -1
    return (price - pos.entry_price) * pos.size * sign


def _resolve_bracket(pos: Position, o, h, l, cfg: BacktestConfig):
    """Return (reason, base_price) if SL/TP fires on this bar, else None.
    Handles gap-through at the open and the stop/tp intrabar policy."""
    sl, tp = pos.stop_loss, pos.take_profit
    stop_hit = tp_hit = False
    stop_base = tp_base = None

    if pos.direction == "long":
        if sl is not None:
            if o <= sl:            # gapped through the stop at the open
                stop_hit, stop_base = True, o
            elif l <= sl:
                stop_hit, stop_base = True, sl
        if tp is not None:
            if o >= tp:            # gapped through the target at the open
                tp_hit, tp_base = True, o
            elif h >= tp:
                tp_hit, tp_base = True, tp
    else:  # short
        if sl is not None:
            if o >= sl:
                stop_hit, stop_base = True, o
            elif h >= sl:
                stop_hit, stop_base = True, sl
        if tp is not None:
            if o <= tp:
                tp_hit, tp_base = True, o
            elif l <= tp:
                tp_hit, tp_base = True, tp

    if stop_hit and tp_hit:
        if cfg.intrabar == "tp_first" or cfg.intrabar == "optimistic":
            return ("tp", tp_base)
        return ("stop", stop_base)          # "stop_first" default (pessimistic)
    if stop_hit:
        return ("stop", stop_base)
    if tp_hit:
        return ("tp", tp_base)
    return None


def _make_trade(pos, exit_time, exit_fill, pnl, reason, exit_index) -> Trade:
    r = pnl / pos.initial_risk if pos.initial_risk not in (None, 0) else None
    return Trade(
        entry_time=pos.entry_time, exit_time=exit_time, direction=pos.direction,
        size=pos.size, entry_price=pos.entry_price, exit_price=exit_fill,
        stop_loss=pos.stop_loss, take_profit=pos.take_profit, pnl=pnl, r_multiple=r,
        exit_reason=reason, tag=pos.tag, bars_held=exit_index - pos.entry_index,
        initial_risk=pos.initial_risk,
    )


def _slice_htf(htf, current_time):
    # Only HTF bars that have fully closed (right-labeled close <= now) -> no look-ahead.
    return {k: df[df.index <= current_time] for k, df in htf.items()}


def run_backtest(config: BacktestConfig, strategy: Strategy, data: pd.DataFrame,
                 htf: Optional[dict] = None) -> BacktestResult:
    cfg = config
    htf = htf or {}
    balance = cfg.opening_balance
    equity = balance
    position: Optional[Position] = None
    pending_order = None
    pending_close = False
    pending_close_reason = "manual"
    trades: list[Trade] = []
    rows = []
    stopped = False

    times = data.index
    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    n = len(data)

    strategy.on_start(Context(data, 0, None, equity, balance, _slice_htf(htf, times[0])))

    i = 0
    for i in range(n):
        o, h, l, c, t = opens[i], highs[i], lows[i], closes[i], times[i]

        # 1) discretionary close requested last bar fills at THIS open (first event of bar)
        if position is not None and pending_close:
            base = o
            exit_fill = _sell_fill(base, cfg) if position.direction == "long" else _buy_fill(base, cfg)
            pnl = _trade_pnl(position, exit_fill, cfg)
            balance += pnl
            trades.append(_make_trade(position, t, exit_fill, pnl, pending_close_reason, i))
            position = None
        pending_close = False

        # 2) manage existing position: SL/TP may fire intrabar
        if position is not None:
            hit = _resolve_bracket(position, o, h, l, cfg)
            if hit is not None:
                reason, base = hit
                exit_fill = _sell_fill(base, cfg) if position.direction == "long" else _buy_fill(base, cfg)
                pnl = _trade_pnl(position, exit_fill, cfg)
                balance += pnl
                trades.append(_make_trade(position, t, exit_fill, pnl, reason, i))
                position = None

        # 3) execute pending entry at THIS open, if flat
        if pending_order is not None and position is None:
            order = pending_order
            entry_fill = _buy_fill(o, cfg) if order.direction == "long" else _sell_fill(o, cfg)
            init_risk = (abs(entry_fill - order.stop_loss) * order.size
                         if order.stop_loss is not None else None)
            position = Position(order.direction, order.size, entry_fill, t, i,
                                order.stop_loss, order.take_profit, init_risk, order.tag)
        pending_order = None

        # 4) mark-to-market at this close (so ctx.equity is current for sizing)
        equity = balance + _unrealized(position, c)

        # 5) ask strategy for decisions using data up to this bar
        ctx = Context(data, i, position, equity, balance, _slice_htf(htf, t))
        strategy.on_bar(ctx)
        if ctx._order is not None and position is None:
            pending_order = ctx._order         # pyramiding not supported: ignore if in position
        if ctx._close_requested and position is not None:
            pending_close = True
            pending_close_reason = ctx._close_reason

        rows.append((t, equity, balance))

        if equity <= 0:
            stopped = True
            break

    strategy.on_finish(Context(data, i, position, equity, balance, _slice_htf(htf, times[i])))

    ec = pd.DataFrame(rows, columns=["time", "equity", "balance"]).set_index("time")
    peak = ec["equity"].cummax()
    ec["peak"] = peak
    ec["drawdown"] = ec["equity"] - peak
    ec["drawdown_pct"] = ec["drawdown"] / peak

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    diffs = np.diff(times.asi8) / 1e9 if n > 1 else np.array([0.0])
    tf_seconds = float(np.median(diffs)) if n > 1 else 0.0

    return BacktestResult(
        equity_curve=ec, trades=trades_df, config=cfg,
        data_start=times[0], data_end=times[-1],
        timeframe_seconds=tf_seconds, stopped_out=stopped,
    )
