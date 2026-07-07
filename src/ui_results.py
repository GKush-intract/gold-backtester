from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def render_param_inputs(strat_cls, key_prefix: str = "") -> dict:
    """Render number/checkbox/text inputs for a Strategy subclass's params schema.
    Returns {param_name: value}. key_prefix keeps widget keys unique across pages."""
    param_values = {}
    for pname, spec in strat_cls.params.items():
        ptype, default, pmin, pmax, help_ = spec
        key = f"{key_prefix}{pname}"
        if ptype == "int":
            param_values[pname] = st.number_input(pname, int(pmin), int(pmax), int(default),
                                                   step=1, help=help_, key=key)
        elif ptype == "float":
            param_values[pname] = st.number_input(pname, float(pmin), float(pmax),
                                                   float(default), help=help_, key=key)
        elif ptype == "bool":
            param_values[pname] = st.checkbox(pname, bool(default), help=help_, key=key)
        else:
            param_values[pname] = st.text_input(pname, str(default), help=help_, key=key)
    return param_values


def render_results(res, m, summary, cfg, elapsed: float, num_bars: int) -> None:
    """Metrics row, equity/drawdown chart, R histogram, trade log, downloads.
    Shared by the classic backtester page and the Strategy Builder page."""
    st.success(f"Done in {elapsed:.2f}s — {num_bars} bars, {m['num_trades']} trades"
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
