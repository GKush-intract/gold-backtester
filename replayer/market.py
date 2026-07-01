from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path("data/cache/dukascopy")
DEFAULT_M1 = CACHE / "XAUUSD_m1_20240101_20240401.parquet"


class MarketFeed:
    """A causal cursor over m1 bars. Serves the full history for chart bootstrap and streams
    one bar at a time as the replay clock advances. Never exposes future bars past the cursor."""

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._t = df["timestamp"].dt.as_unit("ms").astype("int64").to_numpy()  # epoch ms
        self._o = df["open"].to_numpy(float)
        self._h = df["high"].to_numpy(float)
        self._l = df["low"].to_numpy(float)
        self._c = df["close"].to_numpy(float)
        self._v = df["volume"].to_numpy(float)
        self._i = 0  # cursor = index of the NEXT bar to emit

    @classmethod
    def from_parquet(cls, path=DEFAULT_M1) -> "MarketFeed":
        return cls(pd.read_parquet(path))

    def __len__(self):
        return len(self._t)

    @property
    def at_end(self) -> bool:
        return self._i >= len(self._t)

    @property
    def cursor(self) -> int:
        return self._i

    def _bar(self, i) -> dict:
        return {"t": int(self._t[i]), "o": float(self._o[i]), "h": float(self._h[i]),
                "l": float(self._l[i]), "c": float(self._c[i]), "v": float(self._v[i])}

    def peek(self) -> dict | None:
        return None if self.at_end else self._bar(self._i)

    def next_bar(self) -> dict | None:
        if self.at_end:
            return None
        bar = self._bar(self._i)
        self._i += 1
        return bar

    def seek(self, ts_ms: int) -> None:
        """Position the cursor at the first bar with t >= ts_ms."""
        import numpy as np
        self._i = int(np.searchsorted(self._t, ts_ms, side="left"))

    def skip_gap(self, threshold_ms: int) -> bool:
        """Return True if the time from the previously emitted bar to the next bar exceeds
        threshold_ms (a market-closed gap the caller may want to jump the clock across)."""
        if self.at_end or self._i == 0:
            return False
        prev_t = int(self._t[self._i - 1])
        return (int(self._t[self._i]) - prev_t) > threshold_ms

    def candles(self, upto_cursor: bool = False) -> list[dict]:
        """All bars as JSON-ready dicts. upto_cursor=True returns only already-emitted bars."""
        end = self._i if upto_cursor else len(self._t)
        return [self._bar(i) for i in range(end)]
