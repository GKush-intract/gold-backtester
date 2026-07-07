from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from src import runner
from src.data_fetcher import fetch_ohlc
from src.data_loader import load_ohlc, resample
from src.engine import BacktestConfig
from src.metrics import compute_metrics
from src.strategies import get_strategy_registry
from src.ui_results import render_param_inputs, render_results

st.set_page_config(page_title="Gold Backtester", layout="wide")
st.title("🥇 Gold Backtesting Platform")

registry = get_strategy_registry()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Fetch from Dukascopy", "Local CSV", "Synthetic"])
    timeframe = st.selectbox("Base timeframe", ["m1", "m5", "m15", "m30", "h1", "h4"], index=1)
    start = st.date_input("Start", dt.date(2024, 1, 2))
    end = st.date_input("End", dt.date(2024, 2, 1))
    csv_path = None
    if source == "Local CSV":
        files = [p.name for p in Path("data/raw").glob("*.csv")]
        csv_path = st.selectbox("CSV file", files) if files else None
        if not files:
            st.warning("No CSVs in data/raw/")
    resample_to = st.selectbox("Resample to (optional)",
                               ["(none)", "m5", "m15", "m30", "h1", "h4"], index=0)

    st.header("Strategy")
    strat_name = st.selectbox("Strategy", list(registry.keys()))
    strat_cls = registry[strat_name]
    param_values = render_param_inputs(strat_cls)

    st.header("Account & costs")
    balance = st.number_input("Opening balance", 100.0, 1e9, 10_000.0, step=1000.0)
    spread = st.number_input("Spread (price units)", 0.0, 10.0, 0.30)
    slippage = st.number_input("Slippage (price units)", 0.0, 10.0, 0.0)
    commission_trade = st.number_input("Commission per trade ($)", 0.0, 1000.0, 0.0)
    commission_lot = st.number_input("Commission per lot ($/lot)", 0.0, 1000.0, 0.0,
                                     help="1 lot = 100 oz")
    max_leverage = st.number_input("Max leverage (×)", 0.0, 500.0, 20.0,
                                   help="Caps position notional at this × equity; 0 = unlimited")
    intrabar = st.selectbox("Intrabar SL/TP rule", ["stop_first", "tp_first", "optimistic"])
    st.caption("Size is in **lots** (1 lot = 100 oz). The trade log shows notional & leverage.")
    run = st.button("▶ Run backtest", type="primary")

# ---------------- Run ----------------
if run:
    with st.spinner("Loading data..."):
        if source == "Fetch from Dukascopy":
            raw = fetch_ohlc("XAUUSD", timeframe,
                             dt.datetime.combine(start, dt.time()),
                             dt.datetime.combine(end, dt.time()))
            data = load_ohlc(raw)
        elif source == "Local CSV" and csv_path:
            data = load_ohlc(str(Path("data/raw") / csv_path),
                             start=str(start), end=str(end))
        elif source == "Synthetic":
            import numpy as np
            idx = pd.date_range(start, end, freq="5min", tz="UTC")
            rng = np.random.default_rng(7)
            base = 2000 + rng.normal(0, 1.5, len(idx)).cumsum()
            data = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                                 "close": base + rng.normal(0, .5, len(idx)), "volume": 1.0},
                                index=idx)
            data["high"] = data[["open", "high", "close"]].max(axis=1)
            data["low"] = data[["open", "low", "close"]].min(axis=1)
            data.index.name = "timestamp"
            data = load_ohlc(data.reset_index())
        else:
            st.error("No data source selected / CSV missing.")
            st.stop()

        if resample_to != "(none)":
            data = resample(data, resample_to)

    cfg = BacktestConfig(opening_balance=balance, spread=spread, slippage=slippage,
                         commission_per_trade=commission_trade,
                         commission_per_lot=commission_lot, max_leverage=max_leverage,
                         intrabar=intrabar)
    strat = strat_cls(**param_values)

    t0 = dt.datetime.now()
    res = runner.run_backtest(cfg, strat, data)
    elapsed = (dt.datetime.now() - t0).total_seconds()
    m, summary = compute_metrics(res)

    render_results(res, m, summary, cfg, elapsed, len(data))
else:
    st.info("Configure the run in the sidebar and click **Run backtest**.")
