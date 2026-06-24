from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from src import runner
from src.data_fetcher import fetch_ohlc
from src.data_loader import load_ohlc
from src.engine import BacktestConfig
from src.metrics import compute_metrics
from src.strategies import get_strategy_registry


def _synthetic(n=500):
    idx = pd.date_range("2024-01-02", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1.5, n).cumsum()
    base = 2000 + steps
    df = pd.DataFrame({
        "open": base, "high": base + np.abs(rng.normal(0, 1, n)),
        "low": base - np.abs(rng.normal(0, 1, n)), "close": base + rng.normal(0, 0.5, n),
        "volume": 1.0,
    }, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df.index.name = "timestamp"
    return df.reset_index()


def main():
    ap = argparse.ArgumentParser(description="Headless gold backtest")
    ap.add_argument("--synthetic", action="store_true", help="use generated random-walk data")
    ap.add_argument("--csv", help="path to OHLC CSV in data/raw/")
    ap.add_argument("--fetch", action="store_true", help="fetch XAUUSD from Dukascopy")
    ap.add_argument("--timeframe", default="m5")
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--end", default="2024-02-01")
    ap.add_argument("--strategy", default="MA Crossover")
    ap.add_argument("--balance", type=float, default=10_000)
    args = ap.parse_args()

    if args.synthetic:
        data = load_ohlc(_synthetic())
    elif args.fetch:
        start = dt.datetime.fromisoformat(args.start)
        end = dt.datetime.fromisoformat(args.end)
        data = load_ohlc(fetch_ohlc("XAUUSD", args.timeframe, start, end))
    elif args.csv:
        data = load_ohlc(args.csv)
    else:
        raise SystemExit("pick one of --synthetic / --fetch / --csv")

    reg = get_strategy_registry()
    if args.strategy not in reg:
        raise SystemExit(f"unknown strategy {args.strategy!r}; available: {list(reg)}")
    strat = reg[args.strategy]()

    res = runner.run_backtest(BacktestConfig(opening_balance=args.balance), strat, data)
    m, summary = compute_metrics(res)
    print(f"\nStrategy: {args.strategy} | bars: {len(data)} | trades: {m['num_trades']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
