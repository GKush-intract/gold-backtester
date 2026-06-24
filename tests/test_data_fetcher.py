import datetime as dt
import pandas as pd
import pytest
from src import data_fetcher


def _fake_df():
    idx = pd.to_datetime(["2024-01-02 00:00", "2024-01-02 00:05"], utc=True)
    return pd.DataFrame(
        {"timestamp": idx, "open": [2000.0, 2001.0], "high": [2002.0, 2003.0],
         "low": [1999.0, 2000.5], "close": [2001.0, 2002.5], "volume": [1.0, 2.0]}
    )


def test_fetch_writes_then_reads_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_download(symbol, timeframe, start, end):
        calls["n"] += 1
        return _fake_df()

    monkeypatch.setattr(data_fetcher, "_download", fake_download)
    start, end = dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)

    df1 = data_fetcher.fetch_ohlc("XAUUSD", "m5", start, end)
    df2 = data_fetcher.fetch_ohlc("XAUUSD", "m5", start, end)

    assert calls["n"] == 1  # second call served from cache, no download
    assert list(df1.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df2) == 2


def test_unsupported_timeframe_raises():
    with pytest.raises(ValueError):
        data_fetcher.fetch_ohlc("XAUUSD", "m7", dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3))


def test_missing_dates_raise():
    with pytest.raises(ValueError):
        data_fetcher.fetch_ohlc("XAUUSD", "m5", None, None)
