from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class Order:
    direction: str                     # "long" | "short"
    size: float                        # units (oz), > 0
    stop_loss: Optional[float] = None  # absolute price
    take_profit: Optional[float] = None
    tag: str = ""


@dataclass
class Position:
    direction: str
    size: float
    entry_price: float
    entry_time: Any
    entry_index: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    initial_risk: Optional[float]      # abs(entry_fill - stop) * size, set at fill
    tag: str = ""


class Context:
    """Per-bar read-only view of state + order API. Records intent; engine executes it."""

    def __init__(self, data, index, position, equity, balance, htf):
        self.data = data
        self.index = index
        self.position = position
        self.equity = equity
        self.balance = balance
        self.htf = htf
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
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if size is None or size <= 0:
            return  # ignore non-positive sizing
        self._order = Order(direction, float(size), stop_loss, take_profit, tag)

    def close(self, reason="manual"):
        self._close_requested = True
        self._close_reason = reason

    def size_for_risk(self, risk_pct, entry_price, stop_price) -> float:
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        return (self.equity * risk_pct) / risk_per_unit


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
