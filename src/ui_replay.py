from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

SPANS = [100, 200, 400, 800]  # selectable window sizes (bars)
_PALETTE = ["#f39c12", "#3498db", "#9b59b6", "#16a085", "#e67e22", "#7f8c8d"]


def detect_indicator_defaults(params: dict | None) -> str:
    """Overlay spec seeded from a strategy's params: any base-timeframe *ema* period
    param becomes an EMA overlay (HTF params are excluded — different timeframe)."""
    if not params:
        return ""
    periods = []
    for k, v in params.items():
        kl = k.lower()
        if "ema" in kl and "htf" not in kl:
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if 2 <= n <= 500 and n not in periods:
                periods.append(n)
    return ", ".join(f"ema:{n}" for n in sorted(periods))


def parse_overlays(spec: str) -> list[tuple[str, int]]:
    """'ema:33, sma:50' -> [('ema', 33), ('sma', 50)]. Unknown kinds/bad numbers skipped."""
    out = []
    for part in (spec or "").split(","):
        kind, _, num = part.strip().lower().partition(":")
        kind = kind.strip()
        if kind not in ("ema", "sma"):
            continue
        try:
            n = int(num.strip())
        except ValueError:
            continue
        if 2 <= n <= 1000 and (kind, n) not in out:
            out.append((kind, n))
    return out


def trade_bar_positions(trades: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Copy of the trade log with integer bar positions for entry/exit times."""
    t = trades.copy()
    if t.empty:
        t["entry_idx"] = pd.Series(dtype=int)
        t["exit_idx"] = pd.Series(dtype=int)
        return t
    n = len(index)
    t["entry_idx"] = index.searchsorted(pd.DatetimeIndex(t["entry_time"])).clip(0, n - 1)
    t["exit_idx"] = index.searchsorted(pd.DatetimeIndex(t["exit_time"])).clip(0, n - 1)
    return t


def _num(x) -> bool:
    return x is not None and x == x  # not None and not NaN


def _trade_label(i: int, r) -> str:
    rr = f"{r.r_multiple:+.2f}R" if _num(r.r_multiple) else "?R"
    return (f"#{i}  {r.direction} {pd.Timestamp(r.entry_time):%m-%d %H:%M} "
            f"→ {r.exit_reason} ({r.pnl:+.0f}$, {rr})")


def compute_window(center: int, span: int, n: int,
                   trade_len: int | None = None, pad: int = 20) -> tuple[int, int]:
    """Inclusive [lo, hi] window of ~`span` bars centered on `center`, clamped to the
    data. If a selected trade is longer than span - 2*pad, the window widens so the
    whole trade plus `pad` bars of context on each side stays visible."""
    eff = span if trade_len is None else max(span, trade_len + 2 * pad)
    half = eff // 2
    lo, hi = center - half, center + half
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > n - 1:
        lo -= hi - (n - 1)
        hi = n - 1
    return max(lo, 0), hi


def render_replay(res, data: pd.DataFrame, key_prefix: str = "rp_",
                  params: dict | None = None) -> None:
    """Trade inspector: pick a trade and the chart centers it — context bars before
    entry and after exit, entry/exit markers, SL/TP segments, indicator overlays and
    the equity curve. Scroll with the arrow buttons, the slider, or drag-pan/wheel-zoom
    directly on the chart."""
    trades = trade_bar_positions(res.trades, data.index)
    n = len(data)
    if n == 0:
        st.info("No data to replay.")
        return

    kcen = f"{key_prefix}center"
    kdd = f"{key_prefix}sel"
    kdone = f"{key_prefix}sel_done"

    def _mid(i: int) -> int:
        return int((trades["entry_idx"].iloc[i] + trades["exit_idx"].iloc[i]) // 2)

    if kcen not in st.session_state:
        st.session_state[kcen] = _mid(0) if len(trades) else min(100, n - 1)
        if len(trades):
            st.session_state[kdd] = 0
            st.session_state[kdone] = 0
    st.session_state[kcen] = int(min(max(st.session_state[kcen], 0), n - 1))

    def _select_trade(i: int) -> None:
        i = int(min(max(i, 0), len(trades) - 1))
        st.session_state[kdd] = i
        st.session_state[kdone] = i
        st.session_state[kcen] = _mid(i)

    span_key = f"{key_prefix}span"
    span = st.session_state.get(span_key, 200)

    # buttons render BEFORE the selectbox/slider so they may set widget state this run
    c = st.columns([1.2, 0.7, 0.7, 1.2, 2.7, 0.9])
    if c[0].button("⏮ prev trade", key=f"{key_prefix}pt") and len(trades):
        cur_sel = st.session_state.get(kdd)
        _select_trade((cur_sel - 1) if cur_sel is not None else 0)
    if c[1].button("◀", key=f"{key_prefix}sl", help=f"scroll left {span // 2} bars"):
        st.session_state[kcen] = max(0, st.session_state[kcen] - span // 2)
    if c[2].button("▶", key=f"{key_prefix}sr", help=f"scroll right {span // 2} bars"):
        st.session_state[kcen] = min(n - 1, st.session_state[kcen] + span // 2)
    if c[3].button("next trade ⏭", key=f"{key_prefix}nt") and len(trades):
        cur_sel = st.session_state.get(kdd)
        _select_trade((cur_sel + 1) if cur_sel is not None else 0)

    sel = None
    if len(trades):
        labels = {i: _trade_label(i, r) for i, r in enumerate(trades.itertuples())}
        sel = c[4].selectbox("Trade", options=[None] + list(labels),
                             format_func=lambda i: "(no trade selected)" if i is None else labels[i],
                             key=kdd, label_visibility="collapsed")
        if sel is not None and st.session_state.get(kdone) != sel:
            # picked from the dropdown: recenter (kdd already holds the new value)
            st.session_state[kdone] = sel
            st.session_state[kcen] = _mid(sel)
    span = c[5].selectbox("Window", SPANS, index=1, key=span_key,
                          label_visibility="collapsed",
                          help="Window size in bars — the trade stays centered")

    center = st.slider("Center bar", 0, n - 1, key=kcen)
    spec = st.text_input("Indicator overlays", detect_indicator_defaults(params),
                         key=f"{key_prefix}ind",
                         help="Comma-separated, computed on the backtest timeframe: "
                              "ema:33, sma:50. (HTF indicators aren't drawn — "
                              "different timeframe.)")
    overlays = parse_overlays(spec)

    trade_len = None
    if sel is not None:
        trade_len = int(trades["exit_idx"].iloc[sel] - trades["entry_idx"].iloc[sel] + 1)
    lo, hi = compute_window(center, span, n, trade_len)
    win = data.iloc[lo:hi + 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=win.index, open=win["open"], high=win["high"],
                                 low=win["low"], close=win["close"], showlegend=False),
                  row=1, col=1)

    for j, (kind, per) in enumerate(overlays):
        if kind == "ema":
            series = data["close"].ewm(span=per, adjust=False).mean()
        else:
            series = data["close"].rolling(per).mean()
        fig.add_trace(go.Scatter(x=win.index, y=series.iloc[lo:hi + 1], mode="lines",
                                 name=f"{kind.upper()}({per})",
                                 line=dict(width=1.2, color=_PALETTE[j % len(_PALETTE)])),
                      row=1, col=1)

    # explicit axis ranges: markers/lines outside the window must not stretch the view
    ymin, ymax = float(win["low"].min()), float(win["high"].max())
    ypad = (ymax - ymin) * 0.08 or 1.0
    fig.update_xaxes(range=[win.index[0], win.index[-1]], row=1, col=1)
    fig.update_yaxes(range=[ymin - ypad, ymax + ypad], row=1, col=1)

    vis = trades[(trades["exit_idx"] >= lo) & (trades["entry_idx"] <= hi)] if len(trades) else trades
    for i, r in zip(vis.index, vis.itertuples()):
        col = "#2ecc71" if r.pnl >= 0 else "#e74c3c"
        hl = (sel is not None and i == sel)
        et, xt = data.index[r.entry_idx], data.index[r.exit_idx]
        if _num(r.stop_loss):
            fig.add_trace(go.Scatter(x=[et, xt], y=[r.stop_loss] * 2, mode="lines",
                                     line=dict(dash="dash", width=2 if hl else 1,
                                               color="#e74c3c"),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)
        if _num(r.take_profit):
            fig.add_trace(go.Scatter(x=[et, xt], y=[r.take_profit] * 2, mode="lines",
                                     line=dict(dash="dash", width=2 if hl else 1,
                                               color="#2ecc71"),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[et, xt], y=[r.entry_price, r.exit_price], mode="lines",
                                 line=dict(dash="dot", width=2 if hl else 1, color=col),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[et], y=[r.entry_price], mode="markers",
            marker=dict(symbol="triangle-up" if r.direction == "long" else "triangle-down",
                        size=16 if hl else 11,
                        color="#2ecc71" if r.direction == "long" else "#e74c3c",
                        line=dict(width=1, color="#333")),
            showlegend=False,
            hovertext=f"entry {r.direction} @{r.entry_price:.2f} [{r.tag}] size {r.size}",
            hoverinfo="text"), row=1, col=1)
        rr = f"{r.r_multiple:+.2f}R" if _num(r.r_multiple) else "?R"
        fig.add_trace(go.Scatter(
            x=[xt], y=[r.exit_price], mode="markers+text",
            marker=dict(symbol="x", size=15 if hl else 10, color=col),
            text=[r.exit_reason], textposition="top center",
            textfont=dict(size=10 if hl else 8),
            showlegend=False,
            hovertext=f"exit @{r.exit_price:.2f} {r.pnl:+.0f}$ {rr} ({r.exit_reason})",
            hoverinfo="text"), row=1, col=1)

    eq = res.equity_curve
    eqw = eq.iloc[lo:min(hi + 1, len(eq))]
    fig.add_trace(go.Scatter(x=eqw.index, y=eqw["equity"], mode="lines",
                             line=dict(width=1.3), showlegend=False), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=560, dragmode="pan",
                      margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False})

    if sel is not None:
        r = trades.iloc[sel]
        rr = f"{r['r_multiple']:+.2f}R" if _num(r["r_multiple"]) else "?R"
        st.caption(f"**Trade #{sel}** — {r['direction']} {r['size']} lots | "
                   f"entry {pd.Timestamp(r['entry_time']):%Y-%m-%d %H:%M} @{r['entry_price']:.2f} → "
                   f"exit {pd.Timestamp(r['exit_time']):%Y-%m-%d %H:%M} @{r['exit_price']:.2f} "
                   f"({r['exit_reason']}, {r['pnl']:+.0f}$, {rr}, {int(r['bars_held'])} bars) | "
                   f"window bars {lo}–{hi} of {n}")
    else:
        bar = data.iloc[center]
        st.caption(f"**Bar {center}/{n - 1}** — {data.index[center]:%Y-%m-%d %H:%M} UTC | "
                   f"O {bar['open']:.2f} H {bar['high']:.2f} "
                   f"L {bar['low']:.2f} C {bar['close']:.2f}")
