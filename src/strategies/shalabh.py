from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import Strategy, OZ_PER_LOT, clamp_lots

# IST trading session for the time filter.
_SESSION_START_MIN = 9 * 60 + 30   # 09:30 IST
_SESSION_END_MIN = 14 * 60 + 30    # 14:30 IST


def _wilder_adx(high, low, close, n):
    """Wilder ADX (matches ta.dmi's adx output closely). Causal: adx[i] uses bars <= i.
    Returns an array of len(close) with NaN during warmup."""
    L = len(close)
    adx = np.full(L, np.nan)
    if L <= 2 * n:
        return adx

    tr = np.zeros(L)
    plus_dm = np.zeros(L)
    minus_dm = np.zeros(L)
    for i in range(1, L):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = np.zeros(L)
    pdm = np.zeros(L)
    mdm = np.zeros(L)
    dx = np.full(L, np.nan)

    # Wilder running sums seeded at index n (sum of bars 1..n).
    atr[n] = tr[1:n + 1].sum()
    pdm[n] = plus_dm[1:n + 1].sum()
    mdm[n] = minus_dm[1:n + 1].sum()
    for i in range(n + 1, L):
        atr[i] = atr[i - 1] - atr[i - 1] / n + tr[i]
        pdm[i] = pdm[i - 1] - pdm[i - 1] / n + plus_dm[i]
        mdm[i] = mdm[i - 1] - mdm[i - 1] / n + minus_dm[i]

    for i in range(n, L):
        if atr[i] == 0:
            dx[i] = 0.0
        else:
            pdi = 100 * pdm[i] / atr[i]
            mdi = 100 * mdm[i] / atr[i]
            s = pdi + mdi
            dx[i] = 100 * abs(pdi - mdi) / s if s > 0 else 0.0

    # ADX = Wilder average of DX; first value at index 2n averages dx[n+1 .. 2n].
    adx[2 * n] = np.nanmean(dx[n + 1:2 * n + 1])
    for i in range(2 * n + 1, L):
        adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx


class ShalabhStrategy(Strategy):
    """VWAP + EMA PRO intraday strategy (ported from PineScript v6).

    Long when: EMA_fast > EMA_slow, close > EMA_trend, close > VWAP, this bar pulled back to
    VWAP (low <= VWAP) and reclaimed it, bullish candle, ADX > threshold, inside the IST session.
    Short is the mirror. Exits: close crossing back through VWAP, or a trailing stop.

    Engine-fit notes: the trailing stop is engine-managed intrabar (trail distance = trail_pct of
    entry price, attached at entry). The VWAP-cross exit is evaluated on bar close and fills next
    open. VWAP is daily-anchored to the IST day using close*volume (tick volume). One position at
    a time. NOTE: TradingView's trail_points/trail_offset are in TICKS, so the original Pine ran a
    far tighter (~$0.10) intrabar trail than the 0.5% intended here — see the design notes.
    """

    name = "Shalabh's strategy"
    params = {
        "ema_fast": ("int", 9, 2, 200, "EMA Fast length"),
        "ema_slow": ("int", 15, 2, 300, "EMA Slow length"),
        "ema_trend": ("int", 200, 10, 600, "EMA Trend length"),
        "adx_len": ("int", 14, 2, 100, "ADX length"),
        "adx_thresh": ("float", 20.0, 0.0, 100.0, "ADX threshold"),
        "use_time_filter": ("bool", True, None, None, "Use IST 09:30–14:30 session filter"),
        "trail_pct": ("float", 0.5, 0.0, 10.0, "Trailing stop %"),
        "equity_pct": ("float", 100.0, 1.0, 100.0, "Position size as % of equity (notional)"),
    }

    def on_start(self, ctx):
        df = ctx.data
        close = df["close"]
        self._ema_fast = close.ewm(span=self.p["ema_fast"], adjust=False).mean().to_numpy()
        self._ema_slow = close.ewm(span=self.p["ema_slow"], adjust=False).mean().to_numpy()
        self._ema_trend = close.ewm(span=self.p["ema_trend"], adjust=False).mean().to_numpy()
        self._vwap = self._daily_vwap(df)
        self._adx = _wilder_adx(df["high"].to_numpy(float), df["low"].to_numpy(float),
                                close.to_numpy(float), self.p["adx_len"])
        self._warmup = max(self.p["ema_trend"], 2 * self.p["adx_len"] + 1)

    @staticmethod
    def _daily_vwap(df):
        """VWAP = cumsum(close*volume)/cumsum(volume), reset each IST calendar day."""
        ist_day = df.index.tz_convert("Asia/Kolkata").floor("D")
        pv = pd.Series(df["close"].to_numpy() * df["volume"].to_numpy(), index=df.index)
        cum_pv = pv.groupby(ist_day).cumsum()
        cum_v = df["volume"].groupby(ist_day).cumsum()
        vwap = cum_pv / cum_v.replace(0, np.nan)
        return vwap.ffill().fillna(df["close"]).to_numpy()

    def _in_session(self, t) -> bool:
        if not self.p["use_time_filter"]:
            return True
        ist = t.tz_convert("Asia/Kolkata")
        minutes = ist.hour * 60 + ist.minute
        return _SESSION_START_MIN <= minutes <= _SESSION_END_MIN

    def _size(self, ctx, price) -> float:
        notional = ctx.equity * self.p["equity_pct"] / 100.0
        lots = notional / (price * OZ_PER_LOT)
        return clamp_lots(lots, ctx.equity, price, ctx.max_leverage)

    def on_bar(self, ctx):
        i = ctx.index
        if i < self._warmup:
            return

        bar = ctx.bar
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        ema_f, ema_s, ema_t = self._ema_fast[i], self._ema_slow[i], self._ema_trend[i]
        vwap = self._vwap[i]
        adx = self._adx[i]

        # --- manage an open position: VWAP-cross exit (the trailing stop is engine-managed
        #     intrabar from the trail distance attached at entry) ---
        if ctx.position is not None:
            if ctx.position.direction == "long" and c < vwap:
                ctx.close("vwap_exit")
            elif ctx.position.direction == "short" and c > vwap:
                ctx.close("vwap_exit")
            return

        if not self._in_session(bar["time"]):
            return
        strong_trend = (not np.isnan(adx)) and adx > self.p["adx_thresh"]
        if not strong_trend:
            return

        long_cond = (ema_f > ema_s and c > ema_t          # trend
                     and c > vwap                          # above VWAP
                     and l <= vwap and c > vwap            # pulled back to & reclaimed VWAP
                     and c > o)                            # bullish candle
        short_cond = (ema_f < ema_s and c < ema_t
                      and c < vwap
                      and h >= vwap and c < vwap
                      and c < o)

        trail = c * self.p["trail_pct"] / 100.0  # trailing-stop distance in price ($)
        if long_cond:
            ctx.enter("long", self._size(ctx, c), tag="BUY", trail=trail)
        elif short_cond:
            ctx.enter("short", self._size(ctx, c), tag="SELL", trail=trail)
