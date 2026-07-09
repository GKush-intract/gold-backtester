from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SPANS = [100, 200, 400, 800]        # initial visible window sizes (bars)
MAX_BARS = 30_000                   # cap on candles embedded in the chart payload
_PALETTE = ["#f39c12", "#3498db", "#9b59b6", "#16a085", "#e67e22", "#7f8c8d"]
_LIB = Path(__file__).parent / "static" / "lightweight_charts_v4.js"


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


def _num(x) -> bool:
    return x is not None and x == x  # not None and not NaN


def _trade_label(i: int, r) -> str:
    rr = f"{r.r_multiple:+.2f}R" if _num(r.r_multiple) else "?R"
    return (f"#{i}  {r.direction} {pd.Timestamp(r.entry_time):%m-%d %H:%M} "
            f"→ {r.exit_reason} ({r.pnl:+.0f}$, {rr})")


def _epoch_s(index: pd.DatetimeIndex) -> pd.Index:
    """Unix seconds, resolution-independent (pandas 3.0 may use us indexes)."""
    return (index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")


def build_chart_payload(res, data: pd.DataFrame, overlays: list[tuple[str, int]],
                        sel: int | None = None, span: int = 200,
                        max_bars: int = MAX_BARS) -> dict:
    """Everything the embedded lightweight-charts needs, as one JSON-able dict:
    candles, trade markers, SL/TP segments for the selected trade, indicator
    overlay series, equity series and the initial visible range."""
    trades = trade_bar_positions(res.trades, data.index)
    n = len(data)

    lo, hi = 0, n - 1
    if n > max_bars:  # huge datasets: embed a window around the selection
        center = int(trades["entry_idx"].iloc[sel]) if sel is not None and len(trades) \
            else n - max_bars // 2
        lo, hi = compute_window(center, max_bars, n)
    sub = data.iloc[lo:hi + 1]
    times = _epoch_s(sub.index)
    epoch_at = lambda idx: int(times[idx - lo])  # bar index -> unix seconds

    candles = [{"time": int(t), "open": round(o, 3), "high": round(h, 3),
                "low": round(l, 3), "close": round(c, 3)}
               for t, o, h, l, c in zip(times, sub["open"], sub["high"],
                                        sub["low"], sub["close"])]

    markers = []
    vis = trades[(trades["exit_idx"] >= lo) & (trades["entry_idx"] <= hi)] if len(trades) else trades
    for i, r in zip(vis.index, vis.itertuples()):
        hl = sel is not None and i == sel
        rr = f"{r.r_multiple:+.1f}R" if _num(r.r_multiple) else "?R"
        if lo <= r.entry_idx <= hi:
            markers.append({
                "time": epoch_at(r.entry_idx),
                "position": "belowBar",
                "shape": "arrowUp" if r.direction == "long" else "arrowDown",
                "color": "#2ecc71" if r.direction == "long" else "#e74c3c",
                "size": 2 if hl else 1,
                "text": f"#{i} in" if hl else "",
            })
        if lo <= r.exit_idx <= hi:
            markers.append({
                "time": epoch_at(r.exit_idx),
                "position": "aboveBar",
                "shape": "circle",
                "color": "#2ecc71" if r.pnl >= 0 else "#e74c3c",
                "size": 2 if hl else 1,
                "text": f"{r.exit_reason} {rr}" if hl else "",
            })
    markers.sort(key=lambda m: m["time"])

    sl_seg, tp_seg, conn = [], [], []
    if sel is not None and len(trades) and sel in set(vis.index):
        r = trades.iloc[sel]
        e_idx = int(max(r["entry_idx"], lo))
        x_idx = int(min(r["exit_idx"], hi))
        t0, t1 = epoch_at(e_idx), epoch_at(x_idx)
        if t1 == t0 and x_idx + 1 <= hi:  # same-bar entry/exit: widen so segments render
            t1 = epoch_at(x_idx + 1)
        if _num(r["stop_loss"]):
            sl_seg = [{"time": t0, "value": round(float(r["stop_loss"]), 3)},
                      {"time": t1, "value": round(float(r["stop_loss"]), 3)}]
        if _num(r["take_profit"]):
            tp_seg = [{"time": t0, "value": round(float(r["take_profit"]), 3)},
                      {"time": t1, "value": round(float(r["take_profit"]), 3)}]
        conn = [{"time": t0, "value": round(float(r["entry_price"]), 3)},
                {"time": t1, "value": round(float(r["exit_price"]), 3)}]

    overlay_series = []
    for j, (kind, per) in enumerate(overlays):
        if kind == "ema":
            series = data["close"].ewm(span=per, adjust=False).mean()
        else:
            series = data["close"].rolling(per).mean()
        seg = series.iloc[lo:hi + 1]
        overlay_series.append({
            "name": f"{kind.upper()}({per})",
            "color": _PALETTE[j % len(_PALETTE)],
            "data": [{"time": int(t), "value": round(float(v), 3)}
                     for t, v in zip(times, seg) if v == v],
        })

    eq = res.equity_curve
    eqw = eq.iloc[lo:min(hi + 1, len(eq))]
    equity = [{"time": int(t), "value": round(float(v), 2)}
              for t, v in zip(_epoch_s(eqw.index), eqw["equity"])]

    # logical (bar-index) range: robust against weekend/session gaps, unlike
    # time-based setVisibleRange
    logical = None
    if len(trades) and sel is not None:
        e_idx, x_idx = int(trades["entry_idx"].iloc[sel]), int(trades["exit_idx"].iloc[sel])
        wlo, whi = compute_window((e_idx + x_idx) // 2, span, n, x_idx - e_idx + 1)
        wlo, whi = max(wlo, lo), min(whi, hi)
        logical = {"from": wlo - lo - 0.5, "to": whi - lo + 0.5}

    return {"candles": candles, "markers": markers, "sl": sl_seg, "tp": tp_seg,
            "conn": conn, "overlays": overlay_series, "equity": equity,
            "logical": logical, "window": [int(lo), int(hi)]}


def _chart_html(payload: dict, height: int = 520) -> str:
    lib = _LIB.read_text()
    return f"""
<div id="chart" style="width:100%;height:{height}px"></div>
<script>{lib}</script>
<script>
const P = {json.dumps(payload)};
const el = document.getElementById('chart');
const chart = LightweightCharts.createChart(el, {{
  height: {height},
  layout: {{ background: {{ color: '#ffffff' }}, textColor: '#333' }},
  grid: {{ vertLines: {{ color: '#f0f0f0' }}, horzLines: {{ color: '#f0f0f0' }} }},
  timeScale: {{ timeVisible: true, secondsVisible: false, rightOffset: 4 }},
  rightPriceScale: {{ scaleMargins: {{ top: 0.05, bottom: 0.22 }} }},
  leftPriceScale: {{ visible: true, scaleMargins: {{ top: 0.8, bottom: 0 }} }},
  crosshair: {{ mode: 0 }},
}});
const candles = chart.addCandlestickSeries({{
  upColor: '#2ecc71', downColor: '#e74c3c',
  wickUpColor: '#2ecc71', wickDownColor: '#e74c3c', borderVisible: false,
}});
candles.setData(P.candles);
candles.setMarkers(P.markers);
for (const ov of P.overlays) {{
  const s = chart.addLineSeries({{ color: ov.color, lineWidth: 1, title: ov.name,
                                   priceLineVisible: false, lastValueVisible: false }});
  s.setData(ov.data);
}}
const seg = (data, color, style, width) => {{
  if (!data || !data.length) return;
  const s = chart.addLineSeries({{ color: color, lineWidth: width, lineStyle: style,
                                   priceLineVisible: false, lastValueVisible: false }});
  s.setData(data);
}};
seg(P.sl, '#e74c3c', 2, 2);      // dashed stop-loss of the selected trade
seg(P.tp, '#2ecc71', 2, 2);      // dashed take-profit
seg(P.conn, '#7f8c8d', 1, 1);    // dotted entry->exit connector
if (P.equity.length) {{
  const eq = chart.addLineSeries({{ priceScaleId: 'left', color: 'rgba(52,152,219,0.85)',
                                    lineWidth: 1, priceLineVisible: false,
                                    lastValueVisible: false, title: 'equity' }});
  eq.setData(P.equity);
}}
const fit = () => chart.applyOptions({{ width: el.clientWidth }});
new ResizeObserver(fit).observe(el);
fit();
if (P.logical) chart.timeScale().setVisibleLogicalRange(P.logical);
else chart.timeScale().fitContent();
</script>
"""


def render_replay(res, data: pd.DataFrame, key_prefix: str = "rp_",
                  params: dict | None = None) -> None:
    """Trade inspector on TradingView lightweight-charts: native drag-scroll and
    wheel-zoom over the whole backtest, entry/exit markers for every trade, SL/TP
    and connector segments for the selected trade (kept centered with context
    before entry and after exit), indicator overlays and the equity curve."""
    trades = trade_bar_positions(res.trades, data.index)
    n = len(data)
    if n == 0:
        st.info("No data to replay.")
        return

    kdd = f"{key_prefix}sel"
    c = st.columns([1.2, 1.2, 3.4, 0.9])
    if c[0].button("⏮ prev trade", key=f"{key_prefix}pt") and len(trades):
        cur = st.session_state.get(kdd)
        st.session_state[kdd] = max((cur - 1) if cur is not None else 0, 0)
    if c[1].button("next trade ⏭", key=f"{key_prefix}nt") and len(trades):
        cur = st.session_state.get(kdd)
        st.session_state[kdd] = min((cur + 1) if cur is not None else 0, len(trades) - 1)

    sel = None
    if len(trades):
        if kdd not in st.session_state:
            st.session_state[kdd] = 0
        labels = {i: _trade_label(i, r) for i, r in enumerate(trades.itertuples())}
        sel = c[2].selectbox("Trade", options=[None] + list(labels),
                             format_func=lambda i: "(no trade — full chart)" if i is None else labels[i],
                             key=kdd, label_visibility="collapsed")
    span = c[3].selectbox("Window", SPANS, index=1, key=f"{key_prefix}span",
                          label_visibility="collapsed",
                          help="Initial zoom in bars — scroll/zoom freely on the chart itself")

    spec = st.text_input("Indicator overlays", detect_indicator_defaults(params),
                         key=f"{key_prefix}ind",
                         help="Comma-separated, computed on the backtest timeframe: "
                              "ema:33, sma:50. (HTF indicators aren't drawn — "
                              "different timeframe.)")
    payload = build_chart_payload(res, data, parse_overlays(spec), sel=sel, span=span)
    components.html(_chart_html(payload), height=540)

    if sel is not None and len(trades):
        r = trades.iloc[sel]
        rr = f"{r['r_multiple']:+.2f}R" if _num(r["r_multiple"]) else "?R"
        st.caption(f"**Trade #{sel}** — {r['direction']} {r['size']} lots | "
                   f"entry {pd.Timestamp(r['entry_time']):%Y-%m-%d %H:%M} @{r['entry_price']:.2f} → "
                   f"exit {pd.Timestamp(r['exit_time']):%Y-%m-%d %H:%M} @{r['exit_price']:.2f} "
                   f"({r['exit_reason']}, {r['pnl']:+.0f}$, {rr}, {int(r['bars_held'])} bars)")
    lo, hi = payload["window"]
    if (lo, hi) != (0, n - 1):
        st.caption(f"Large dataset: chart shows bars {lo}–{hi} of {n} "
                   f"(window follows the selected trade).")
