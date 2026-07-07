"""Assemble a 1-minute XAUUSD OHLCV CSV for the last year from Dukascopy.

Fetches month-sized chunks via src.data_fetcher.fetch_ohlc (each chunk cached to
data/cache/dukascopy/*.parquet, so re-runs only download what's missing), then
concatenates, dedupes and writes data/raw/XAUUSD_m1_1y.csv — same column format
as the existing XAUUSD_m5_5y.csv.

Usage: python -m scripts.build_m1_csv [start YYYY-MM-DD] [end YYYY-MM-DD]
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetcher import fetch_ohlc  # noqa: E402

OUT = Path("data/raw/XAUUSD_m1_1y.csv")


def month_starts(start: dt.datetime, end: dt.datetime):
    cur = dt.datetime(start.year, start.month, 1)
    while cur < end:
        nxt = dt.datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def main() -> None:
    end = dt.datetime.strptime(sys.argv[2], "%Y-%m-%d") if len(sys.argv) > 2 \
        else dt.datetime.combine(dt.date.today(), dt.time())
    start = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d") if len(sys.argv) > 1 \
        else end - dt.timedelta(days=365)

    chunks = []
    for s, e in month_starts(start, end):
        print(f"fetching m1 {s:%Y-%m-%d} .. {e:%Y-%m-%d} ...", flush=True)
        df = fetch_ohlc("XAUUSD", "m1", s, e)
        print(f"  {len(df)} rows", flush=True)
        chunks.append(df)

    df = pd.concat(chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="last")
            .reset_index(drop=True))
    df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, "
          f"{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}, "
          f"{OUT.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
