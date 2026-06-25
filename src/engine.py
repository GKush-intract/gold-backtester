from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import Context, Position, Strategy, OZ_PER_LOT, clamp_lots


@dataclass
class BacktestConfig:
    opening_balance: float = 10_000.0
    spread: float = 0.30
    slippage: float = 0.0
    commission_per_trade: float = 0.0   # flat $ per round-trip
    commission_per_lot: float = 0.0     # $ per lot per round-trip
    max_leverage: float = 20.0          # cap notional at max_leverage * equity; <=0 = unlimited
    intrabar: str = "stop_first"        # "stop_first" | "tp_first" | "optimistic"
    entry_fill: str = "next_open"       # "next_open" only in v1 (kept for forward-compat)


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    size: float                 # LOTS (1 lot = 100 oz)
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
    notional: float             # lots * 100 * entry_price ($ face value)
    leverage: float             # notional / equity at entry


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


def _commission(size_lots, cfg):
    return cfg.commission_per_trade + cfg.commission_per_lot * size_lots


def _trade_pnl(pos: Position, exit_fill: float, cfg: BacktestConfig) -> float:
    sign = 1 if pos.direction == "long" else -1
    gross = (exit_fill - pos.entry_price) * pos.size * OZ_PER_LOT * sign
    return gross - _commission(pos.size, cfg)


def _unrealized(pos: Optional[Position], price: float) -> float:
    if pos is None:
        return 0.0
    sign = 1 if pos.direction == "long" else -1
    return (price - pos.entry_price) * pos.size * OZ_PER_LOT * sign


def _effective_stop(pos: Position):
    """The binding stop for this bar = the tighter of the fixed stop and the trailing level.
    Returns (stop_price | None, is_trailing)."""
    sl, trail_lvl = pos.stop_loss, pos.trail_level
    if sl is not None and trail_lvl is not None:
        if pos.direction == "long":
            return (max(sl, trail_lvl), trail_lvl >= sl)
        return (min(sl, trail_lvl), trail_lvl <= sl)
    if trail_lvl is not None:
        return (trail_lvl, True)
    if sl is not None:
        return (sl, False)
    return (None, False)


def _resolve_bracket(pos: Position, o, h, l, cfg: BacktestConfig):
    """Return (reason, base_price) if a stop / trailing-stop / TP fires on this bar, else None.
    Uses the stop level carried INTO the bar (pessimistic: a retrace can hit it before the bar's
    extreme would advance a trailing stop). Handles gap-through at the open and the intrabar policy."""
    eff_stop, is_trail = _effective_stop(pos)
    tp = pos.take_profit
    stop_hit = tp_hit = False
    stop_base = tp_base = None

    if pos.direction == "long":
        if eff_stop is not None:
            if o <= eff_stop:           # gapped through the stop at the open
                stop_hit, stop_base = True, o
            elif l <= eff_stop:
                stop_hit, stop_base = True, eff_stop
        if tp is not None:
            if o >= tp:                 # gapped through the target at the open
                tp_hit, tp_base = True, o
            elif h >= tp:
                tp_hit, tp_base = True, tp
    else:  # short
        if eff_stop is not None:
            if o >= eff_stop:
                stop_hit, stop_base = True, o
            elif h >= eff_stop:
                stop_hit, stop_base = True, eff_stop
        if tp is not None:
            if o <= tp:
                tp_hit, tp_base = True, o
            elif l <= tp:
                tp_hit, tp_base = True, tp

    stop_reason = "trailing_stop" if is_trail else "stop"
    if stop_hit and tp_hit:
        # With a single bracket, "optimistic" (assume favorable fill) == "tp_first".
        if cfg.intrabar == "tp_first" or cfg.intrabar == "optimistic":
            return ("tp", tp_base)
        return (stop_reason, stop_base)     # "stop_first" default (pessimistic)
    if stop_hit:
        return (stop_reason, stop_base)
    if tp_hit:
        return ("tp", tp_base)
    return None


def _make_trade(pos, exit_time, exit_fill, pnl, reason, exit_index) -> Trade:
    r = pnl / pos.initial_risk if pos.initial_risk not in (None, 0) else None
    notional = pos.size * OZ_PER_LOT * pos.entry_price
    leverage = notional / pos.entry_equity if pos.entry_equity else 0.0
    return Trade(
        entry_time=pos.entry_time, exit_time=exit_time, direction=pos.direction,
        size=pos.size, entry_price=pos.entry_price, exit_price=exit_fill,
        stop_loss=pos.stop_loss, take_profit=pos.take_profit, pnl=pnl, r_multiple=r,
        exit_reason=reason, tag=pos.tag, bars_held=exit_index - pos.entry_index,
        initial_risk=pos.initial_risk, notional=notional, leverage=leverage,
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

    strategy.on_start(Context(data, 0, None, equity, balance, _slice_htf(htf, times[0]),
                              max_leverage=cfg.max_leverage))

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

        # 2b) ratchet the trailing stop using this bar's extreme (only after surviving the bar)
        if position is not None and position.trail is not None:
            if position.direction == "long":
                lvl = h - position.trail
                if position.trail_level is None or lvl > position.trail_level:
                    position.trail_level = lvl
            else:
                lvl = l + position.trail
                if position.trail_level is None or lvl < position.trail_level:
                    position.trail_level = lvl

        # 3) execute pending entry at THIS open, if flat (size capped to max leverage)
        if pending_order is not None and position is None:
            order = pending_order
            entry_fill = _buy_fill(o, cfg) if order.direction == "long" else _sell_fill(o, cfg)
            size = clamp_lots(order.size, balance, entry_fill, cfg.max_leverage)
            if size > 0:
                if order.trail is not None:
                    trail_level = (entry_fill - order.trail if order.direction == "long"
                                   else entry_fill + order.trail)
                else:
                    trail_level = None
                # initial risk: fixed stop if set, else the trailing distance, else undefined
                if order.stop_loss is not None:
                    risk_price = order.stop_loss
                elif trail_level is not None:
                    risk_price = trail_level
                else:
                    risk_price = None
                init_risk = (abs(entry_fill - risk_price) * size * OZ_PER_LOT
                             if risk_price is not None else None)
                position = Position(direction=order.direction, size=size, entry_price=entry_fill,
                                    entry_time=t, entry_index=i, entry_equity=balance,
                                    stop_loss=order.stop_loss, take_profit=order.take_profit,
                                    initial_risk=init_risk, tag=order.tag,
                                    trail=order.trail, trail_level=trail_level)
        pending_order = None

        # 4) mark-to-market at this close (so ctx.equity is current for sizing)
        equity = balance + _unrealized(position, c)

        # 5) ask strategy for decisions using data up to this bar
        ctx = Context(data, i, position, equity, balance, _slice_htf(htf, t),
                      max_leverage=cfg.max_leverage)
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

    strategy.on_finish(Context(data, i, position, equity, balance, _slice_htf(htf, times[i]),
                               max_leverage=cfg.max_leverage))

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
