from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

WINDOW = 240  # bars shown behind the cursor
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


def render_replay(res, data: pd.DataFrame, key_prefix: str = "rp_",
                  params: dict | None = None) -> None:
    """Bar-by-bar replay of a backtest: candlestick window with entry/exit markers,
    SL/TP segments, trade connectors and the equity curve underneath."""
    trades = trade_bar_positions(res.trades, data.index)
    n = len(data)
    if n == 0:
        st.info("No data to replay.")
        return

    kcur = f"{key_prefix}cursor"
    ksel = f"{key_prefix}sel_done"
    if kcur not in st.session_state:
        start = int(trades["entry_idx"].iloc[0]) if len(trades) else min(WINDOW, n - 1)
        st.session_state[kcur] = start
    st.session_state[kcur] = int(min(max(st.session_state[kcur], 0), n - 1))

    def _jump(delta=None, to=None):
        cur = st.session_state[kcur]
        tgt = to if to is not None else cur + delta
        st.session_state[kcur] = int(min(max(tgt, 0), n - 1))

    # controls render BEFORE the slider so they may modify its state this run
    c = st.columns([1.2, 0.8, 0.8, 0.9, 1.2, 3.1])
    if c[0].button("⏮ prev trade", key=f"{key_prefix}pt"):
        prev = trades[trades["entry_idx"] < st.session_state[kcur]]
        if len(prev):
            _jump(to=int(prev["entry_idx"].iloc[-1]))
    if c[1].button("◀ −10", key=f"{key_prefix}m10"):
        _jump(-10)
    if c[2].button("+1 ▶", key=f"{key_prefix}p1"):
        _jump(+1)
    if c[3].button("+10 ▶▶", key=f"{key_prefix}p10"):
        _jump(+10)
    if c[4].button("next trade ⏭", key=f"{key_prefix}nt"):
        nxt = trades[trades["entry_idx"] > st.session_state[kcur]]
        if len(nxt):
            _jump(to=int(nxt["entry_idx"].iloc[0]))
    if len(trades):
        labels = {i: _trade_label(i, r) for i, r in enumerate(trades.itertuples())}
        sel = c[5].selectbox("Jump to trade", options=[None] + list(labels),
                             format_func=lambda i: "…" if i is None else labels[i],
                             key=f"{key_prefix}sel", label_visibility="collapsed")
        if sel is not None and st.session_state.get(ksel) != sel:
            st.session_state[ksel] = sel
            _jump(to=int(trades["entry_idx"].iloc[sel]))

    cur = st.slider("Bar", 0, n - 1, key=kcur)
    spec = st.text_input("Indicator overlays", detect_indicator_defaults(params),
                         key=f"{key_prefix}ind",
                         help="Comma-separated, computed on the backtest timeframe: "
                              "ema:33, sma:50. (HTF indicators aren't drawn — "
                              "different timeframe.)")
    overlays = parse_overlays(spec)
    lo = max(0, cur - WINDOW)
    win = data.iloc[lo:cur + 1]

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
        fig.add_trace(go.Scatter(x=win.index, y=series.iloc[lo:cur + 1], mode="lines",
                                 name=f"{kind.upper()}({per})",
                                 line=dict(width=1.2, color=_PALETTE[j % len(_PALETTE)])),
                      row=1, col=1)

    vis = trades[(trades["exit_idx"] >= lo) & (trades["entry_idx"] <= cur)] if len(trades) else trades
    for r in vis.itertuples():
        col = "#2ecc71" if r.pnl >= 0 else "#e74c3c"
        et = data.index[r.entry_idx]
        open_end = data.index[min(r.exit_idx, cur)]
        # SL / TP segments for the life of the trade (up to the cursor)
        if _num(r.stop_loss):
            fig.add_trace(go.Scatter(x=[et, open_end], y=[r.stop_loss] * 2, mode="lines",
                                     line=dict(dash="dash", width=1, color="#e74c3c"),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)
        if _num(r.take_profit):
            fig.add_trace(go.Scatter(x=[et, open_end], y=[r.take_profit] * 2, mode="lines",
                                     line=dict(dash="dash", width=1, color="#2ecc71"),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)
        if lo <= r.entry_idx <= cur:
            fig.add_trace(go.Scatter(
                x=[et], y=[r.entry_price], mode="markers",
                marker=dict(symbol="triangle-up" if r.direction == "long" else "triangle-down",
                            size=12, color="#2ecc71" if r.direction == "long" else "#e74c3c",
                            line=dict(width=1, color="#333")),
                showlegend=False,
                hovertext=f"entry {r.direction} @{r.entry_price:.2f} [{r.tag}] size {r.size}",
                hoverinfo="text"), row=1, col=1)
        if lo <= r.exit_idx <= cur:
            xt = data.index[r.exit_idx]
            rr = f"{r.r_multiple:+.2f}R" if _num(r.r_multiple) else "?R"
            fig.add_trace(go.Scatter(
                x=[xt], y=[r.exit_price], mode="markers+text",
                marker=dict(symbol="x", size=11, color=col),
                text=[r.exit_reason], textposition="top center", textfont=dict(size=9),
                showlegend=False,
                hovertext=f"exit @{r.exit_price:.2f} {r.pnl:+.0f}$ {rr} ({r.exit_reason})",
                hoverinfo="text"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[et, xt], y=[r.entry_price, r.exit_price], mode="lines",
                                     line=dict(dash="dot", width=1, color=col),
                                     showlegend=False, hoverinfo="skip"), row=1, col=1)

    eq = res.equity_curve
    eqw = eq.iloc[lo:min(cur + 1, len(eq))]
    fig.add_trace(go.Scatter(x=eqw.index, y=eqw["equity"], mode="lines",
                             line=dict(width=1.3), name="equity", showlegend=False),
                  row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=560,
                      margin=dict(t=20, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    st.plotly_chart(fig, use_container_width=True)

    bar = data.iloc[cur]
    line = (f"**Bar {cur}/{n - 1}** — {data.index[cur]:%Y-%m-%d %H:%M} UTC | "
            f"O {bar['open']:.2f} H {bar['high']:.2f} L {bar['low']:.2f} C {bar['close']:.2f}")
    if len(trades):
        open_now = trades[(trades["entry_idx"] <= cur) & (trades["exit_idx"] > cur)]
        if len(open_now):
            r0 = open_now.iloc[0]
            sl = f"{r0.stop_loss:.2f}" if _num(r0.stop_loss) else "—"
            tp = f"{r0.take_profit:.2f}" if _num(r0.take_profit) else "—"
            upnl = (bar["close"] - r0.entry_price) * (1 if r0.direction == "long" else -1) \
                   * r0["size"] * 100
            line += (f" | 📌 open {r0.direction} {r0['size']} lots @{r0.entry_price:.2f} "
                     f"SL {sl} TP {tp} (u-pnl {upnl:+.0f}$)")
    st.caption(line)
