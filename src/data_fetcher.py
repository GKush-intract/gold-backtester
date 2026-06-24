from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache/dukascopy")

# UI/CLI timeframe string -> dukascopy_python INTERVAL_* attribute name.
_INTERVAL_MAP = {
    "m1": "INTERVAL_MIN_1", "m5": "INTERVAL_MIN_5", "m15": "INTERVAL_MIN_15",
    "m30": "INTERVAL_MIN_30", "h1": "INTERVAL_HOUR_1", "h4": "INTERVAL_HOUR_4",
    "d1": "INTERVAL_DAY_1",
}

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _cache_path(symbol: str, timeframe: str, start, end) -> Path:
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    return CACHE_DIR / f"{symbol}_{timeframe}_{s}_{e}.parquet"


def fetch_ohlc(symbol="XAUUSD", timeframe="m5", start=None, end=None, use_cache=True) -> pd.DataFrame:
    """Fetch OHLC candles from Dukascopy, caching to parquet keyed on (symbol, tf, range)."""
    if start is None or end is None:
        raise ValueError("start and end datetimes are required")
    timeframe = timeframe.lower()
    if timeframe not in _INTERVAL_MAP:
        raise ValueError(f"unsupported timeframe {timeframe!r}; options: {list(_INTERVAL_MAP)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, timeframe, start, end)
    if use_cache and path.exists():
        return pd.read_parquet(path)

    df = _download(symbol, timeframe, start, end)
    df.to_parquet(path)
    return df


def _download(symbol: str, timeframe: str, start, end) -> pd.DataFrame:
    """Hit the Dukascopy feed. Isolated so tests can monkeypatch it (no network)."""
    import dukascopy_python as d
    from dukascopy_python import instruments

    interval = getattr(d, _INTERVAL_MAP[timeframe])
    # "XAUUSD" -> INSTRUMENT_FX_METALS_XAU_USD
    instr_const = f"INSTRUMENT_FX_METALS_{symbol[:3]}_{symbol[3:]}"
    instrument = getattr(instruments, instr_const)

    raw = d.fetch(instrument, interval, d.OFFER_SIDE_BID, start, end)
    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    return df[COLUMNS]
