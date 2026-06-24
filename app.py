from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import runner
from src.data_fetcher import fetch_ohlc
from src.data_loader import load_ohlc, resample
from src.engine import BacktestConfig
from src.metrics import compute_metrics
from src.strategies import get_strategy_registry

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
    param_values = {}
    for pname, spec in strat_cls.params.items():
        ptype, default, pmin, pmax, help_ = spec
        if ptype == "int":
            param_values[pname] = st.number_input(pname, int(pmin), int(pmax), int(default),
                                                   step=1, help=help_)
        elif ptype == "float":
            param_values[pname] = st.number_input(pname, float(pmin), float(pmax),
                                                   float(default), help=help_)
        elif ptype == "bool":
            param_values[pname] = st.checkbox(pname, bool(default), help=help_)
        else:
            param_values[pname] = st.text_input(pname, str(default), help=help_)

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

    st.success(f"Done in {elapsed:.2f}s — {len(data)} bars, {m['num_trades']} trades"
               + ("  ⚠️ run stopped: equity hit 0" if res.stopped_out else ""))

    c = st.columns(6)
    c[0].metric("Total return", f"{m['total_return_pct']*100:.1f}%")
    c[1].metric("Profit factor", f"{m['profit_factor']:.2f}")
    c[2].metric("Win rate", f"{m['win_rate']*100:.1f}%")
    c[3].metric("Max drawdown", f"{m['max_drawdown_pct']*100:.1f}%")
    c[4].metric("Expectancy (R)", f"{m['expectancy_r']:.2f}")
    c[5].metric("# Trades", m["num_trades"])

    ec = res.equity_curve
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05, subplot_titles=("Equity / Balance", "Drawdown"))
    fig.add_trace(go.Scatter(x=ec.index, y=ec["equity"], name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ec.index, y=ec["balance"], name="Balance",
                             line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ec.index, y=ec["drawdown_pct"] * 100, name="Drawdown %",
                             fill="tozeroy"), row=2, col=1)
    fig.update_layout(height=520, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1, 2])
    if not res.trades.empty and res.trades["r_multiple"].notna().any():
        rhist = go.Figure(go.Histogram(x=res.trades["r_multiple"].dropna(), nbinsx=30))
        rhist.update_layout(title="R-multiple distribution", height=360)
        left.plotly_chart(rhist, use_container_width=True)
    else:
        left.info("No R data (no trades with stops).")

    right.subheader("Trade log")
    right.dataframe(res.trades, use_container_width=True, height=360)
    if not res.trades.empty:
        right.download_button("⬇ Download trades CSV",
                              res.trades.to_csv(index=False).encode(),
                              file_name="trades.csv", mime="text/csv")

    with st.expander("Full metrics + config"):
        st.dataframe(summary, use_container_width=True)
        st.caption("Sharpe/Sortino are annualized from per-bar equity returns (risk-free = 0). "
                   "On intraday data these are inflated and not comparable to daily Sharpe — "
                   "treat them as relative, not absolute. SL/TP are first checked the bar after entry.")
        st.json(cfg.__dict__)
        st.download_button("⬇ Download equity curve CSV",
                           ec.to_csv().encode(), file_name="equity_curve.csv", mime="text/csv")
else:
    st.info("Configure the run in the sidebar and click **Run backtest**.")
