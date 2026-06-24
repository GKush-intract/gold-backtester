import pandas as pd
import pytest
from src import data_loader


def _good_df():
    idx = pd.to_datetime(
        ["2024-01-02 00:00", "2024-01-02 00:01", "2024-01-02 00:02"], utc=True
    )
    return pd.DataFrame(
        {"timestamp": idx, "open": [2000, 2001, 2002], "high": [2003, 2004, 2005],
         "low": [1999, 2000, 2001], "close": [2001, 2002, 2003], "volume": [1, 1, 1]}
    )


def test_load_sorts_and_indexes_utc():
    df = _good_df().sample(frac=1, random_state=1)  # shuffle rows
    out = data_loader.load_ohlc(df)
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == "UTC"
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_missing_column_raises():
    df = _good_df().drop(columns=["volume"])
    with pytest.raises(ValueError):
        data_loader.load_ohlc(df)


def test_high_below_low_raises():
    df = _good_df()
    df.loc[1, "high"] = df.loc[1, "low"] - 1
    with pytest.raises(ValueError):
        data_loader.load_ohlc(df)


def test_naive_timestamp_requires_tz():
    df = _good_df()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        data_loader.load_ohlc(df)
    out = data_loader.load_ohlc(df, tz="UTC")  # tz supplied -> ok
    assert str(out.index.tz) == "UTC"


def test_resample_m1_to_m5():
    idx = pd.date_range("2024-01-02 00:00", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"open": range(10), "high": [v + 2 for v in range(10)],
         "low": [v - 1 for v in range(10)], "close": [v + 1 for v in range(10)],
         "volume": [1] * 10}, index=idx)
    df.index.name = "timestamp"
    out = data_loader.resample(df, "m5")
    assert out["volume"].iloc[0] == 5
    assert out["high"].iloc[0] == max(v + 2 for v in range(5))
