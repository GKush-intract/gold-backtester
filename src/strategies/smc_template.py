from __future__ import annotations

import datetime as dt

from ..strategy import Strategy

# London/NY killzones (UTC). Adjust to taste.
LONDON = (dt.time(7, 0), dt.time(10, 0))
NEW_YORK = (dt.time(12, 0), dt.time(15, 0))


def _in_session(t, window) -> bool:
    return window[0] <= t.timetz().replace(tzinfo=None) <= window[1]


class SMCTemplate(Strategy):
    """COPY THIS FILE to start a new SMC/ICT strategy. Non-functional as-is.

    Demonstrates the patterns you'll use:
      - HTF bias via ctx.htf["h1"] / ctx.htf["h4"] while trading the base timeframe
      - session filtering off ctx.bar["time"]
      - structural stop (below/above the swing) + R-based TP, risk-% sizing
    Fill in the TODOs and remove the `return` guard in on_bar to activate.
    """

    name = "SMC Template (inactive)"
    htf_timeframes = ["h1", "h4"]   # runner builds these resampled views
    params = {
        "risk_pct": ("float", 0.01, 0.001, 0.05, "Risk per trade"),
        "rr": ("float", 2.0, 0.5, 10.0, "Reward:risk for TP"),
        "swing_lookback": ("int", 20, 3, 200, "Bars to scan for swing high/low"),
    }

    def on_bar(self, ctx):
        return  # TODO: remove once implemented — template must not trade by default

        # --- 1) Higher-timeframe bias -------------------------------------------------
        h1 = ctx.htf.get("h1")
        # TODO: derive bias, e.g. bullish if last closed H1 close > H1 SMA.
        bias = self._htf_bias(h1)  # "long" | "short" | None

        # --- 2) Session filter --------------------------------------------------------
        now = ctx.bar["time"]
        if not (_in_session(now, LONDON) or _in_session(now, NEW_YORK)):
            return

        # --- 3) Structure on the base timeframe --------------------------------------
        swing_hi, swing_lo = self._recent_swings(ctx.history, self.p["swing_lookback"])
        # TODO: detect BOS/CHoCH and your entry trigger here.
        setup_long = bias == "long"   # placeholder condition
        setup_short = bias == "short"

        price = ctx.bar["close"]
        if ctx.position is None and setup_long and swing_lo is not None:
            stop = swing_lo                       # structural stop below the swing
            tp = price + self.p["rr"] * (price - stop)
            size = ctx.size_for_risk(self.p["risk_pct"], price, stop)
            ctx.enter("long", size, stop_loss=stop, take_profit=tp, tag="smc_long")
        elif ctx.position is None and setup_short and swing_hi is not None:
            stop = swing_hi
            tp = price - self.p["rr"] * (stop - price)
            size = ctx.size_for_risk(self.p["risk_pct"], price, stop)
            ctx.enter("short", size, stop_loss=stop, take_profit=tp, tag="smc_short")

    # ----- stubbed helpers: implement for your edge -----
    @staticmethod
    def _htf_bias(htf_df):
        # TODO: return "long"/"short"/None from higher-timeframe structure.
        return None

    @staticmethod
    def _recent_swings(history, lookback):
        # TODO: real swing detection. Placeholder: window extremes.
        window = history.iloc[-lookback:]
        if len(window) < lookback:
            return None, None
        return float(window["high"].max()), float(window["low"].min())
