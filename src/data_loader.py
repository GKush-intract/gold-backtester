from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]

_RESAMPLE_RULE = {
    "m1": "1min", "m5": "5min", "m15": "15min", "m30": "30min",
    "h1": "1h", "h4": "4h", "d1": "1D",
}


def load_ohlc(source, start=None, end=None, tz=None) -> pd.DataFrame:
    """Load OHLC from a CSV path or DataFrame; validate, sort, dedupe, slice. UTC-indexed."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    df = _normalize(df, tz)
    _validate(df)

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df


def _normalize(df: pd.DataFrame, tz) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        if tz is None:
            raise ValueError("naive timestamps; pass tz= to localize")
        ts = ts.dt.tz_localize(tz).dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def _validate(df: pd.DataFrame) -> None:
    if df[PRICE_COLS].isna().any().any():
        raise ValueError("NaNs present in OHLC columns")
    if not df.index.is_monotonic_increasing:
        raise ValueError("timestamps not monotonic after sort (duplicates?)")
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    if (h < l).any():
        raise ValueError("invalid bar: high < low")
    if (h < o).any() or (h < c).any():
        raise ValueError("invalid bar: high < open/close")
    if (l > o).any() or (l > c).any():
        raise ValueError("invalid bar: low > open/close")
    if (df[PRICE_COLS] <= 0).any().any():
        raise ValueError("non-positive prices present")


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Standard OHLC resample, right-labeled, dropping empty (gap) bins."""
    tf = timeframe.lower()
    if tf not in _RESAMPLE_RULE:
        raise ValueError(f"unsupported timeframe {tf!r}; options: {list(_RESAMPLE_RULE)}")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(_RESAMPLE_RULE[tf], label="right", closed="left").agg(agg)
    return out.dropna(subset=["open"])
