from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

# Size is denominated in LOTS. 1 standard lot of gold = 100 troy ounces.
OZ_PER_LOT = 100.0
# Sizing granularity: round DOWN to this lot step (0.01 = micro lot = 1 oz).
LOT_STEP = 0.01


def round_lots(lots: float) -> float:
    """Round a lot size DOWN to the nearest LOT_STEP (never round risk up)."""
    if lots <= 0:
        return 0.0
    steps = math.floor(lots / LOT_STEP + 1e-9)
    return round(steps * LOT_STEP, 2)


def clamp_lots(lots: float, equity: float, price: float, max_leverage: float) -> float:
    """Cap lots so notional (lots * OZ_PER_LOT * price) <= max_leverage * equity, then
    round down to LOT_STEP. max_leverage <= 0 means no cap."""
    if max_leverage and max_leverage > 0 and price > 0 and equity > 0:
        max_lots = (max_leverage * equity) / (OZ_PER_LOT * price)
        lots = min(lots, max_lots)
    return round_lots(lots)


@dataclass
class Order:
    direction: str                     # "long" | "short"
    size: float                        # LOTS (1 lot = 100 oz), > 0
    stop_loss: Optional[float] = None  # absolute price
    take_profit: Optional[float] = None
    tag: str = ""


@dataclass
class Position:
    direction: str
    size: float                        # LOTS
    entry_price: float
    entry_time: Any
    entry_index: int
    entry_equity: float                # account equity at fill (for leverage reporting)
    stop_loss: Optional[float]
    take_profit: Optional[float]
    initial_risk: Optional[float]      # $ risked = abs(entry_fill - stop) * size_lots * OZ_PER_LOT
    tag: str = ""


class Context:
    """Per-bar read-only view of state + order API. Records intent; engine executes it."""

    def __init__(self, data, index, position, equity, balance, htf, max_leverage=0.0):
        self.data = data
        self.index = index
        self.position = position
        self.equity = equity
        self.balance = balance
        self.htf = htf
        self.max_leverage = max_leverage
        self._order: Optional[Order] = None
        self._close_requested: bool = False
        self._close_reason: str = "manual"

    @property
    def bar(self) -> dict:
        row = self.data.iloc[self.index]
        return {
            "time": self.data.index[self.index],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    @property
    def history(self) -> pd.DataFrame:
        """All bars 0..index inclusive (no look-ahead)."""
        return self.data.iloc[: self.index + 1]

    def enter(self, direction, size, stop_loss=None, take_profit=None, tag=""):
        """Request a market entry, filled next bar open. `size` is in LOTS (1 lot = 100 oz)."""
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if size is None or size <= 0:
            return  # ignore non-positive sizing
        self._order = Order(direction, float(size), stop_loss, take_profit, tag)

    def close(self, reason="manual"):
        self._close_requested = True
        self._close_reason = reason

    def size_for_risk(self, risk_pct, entry_price, stop_price) -> float:
        """LOTS to risk `risk_pct` of equity over the stop distance:
        lots = (equity * risk_pct) / (|entry - stop| * OZ_PER_LOT),
        capped by max_leverage and rounded down to LOT_STEP."""
        risk_per_oz = abs(entry_price - stop_price)
        if risk_per_oz <= 0:
            return 0.0
        lots = (self.equity * risk_pct) / (risk_per_oz * OZ_PER_LOT)
        return clamp_lots(lots, self.equity, entry_price, self.max_leverage)


class Strategy:
    name: str = "Unnamed"
    # schema entry: name -> (type, default, min, max, help)
    params: dict = {}
    # higher-timeframe views the strategy wants (e.g. ["h1", "h4"]); built by runner.
    htf_timeframes: list = []

    def __init__(self, **params):
        merged = {k: spec[1] for k, spec in self.params.items()}
        merged.update(params)
        self.p = merged

    def on_start(self, ctx: Context): ...
    def on_bar(self, ctx: Context): ...
    def on_finish(self, ctx: Context): ...
