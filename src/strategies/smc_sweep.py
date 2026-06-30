from __future__ import annotations

import numpy as np

from ..strategy import Strategy


def _wilder_atr(high, low, close, n):
    """Wilder ATR. Causal: atr[i] uses bars <= i. NaN during warmup."""
    L = len(close)
    atr = np.full(L, np.nan)
    if L <= n:
        return atr
    tr = np.zeros(L)
    tr[0] = high[0] - low[0]
    for i in range(1, L):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[n] = tr[1:n + 1].mean()
    for i in range(n + 1, L):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


def _pivots(values, n, kind):
    """Fractal pivot indices: index i is a pivot if values[i] is the strict extreme of the
    window [i-n, i+n] versus the left side and a (weak) extreme versus the right. A pivot at i
    is only *confirmed* n bars later (it needs n right-hand bars), so callers must gate on that
    to stay causal. Returns (idx_array, val_array)."""
    L = len(values)
    idx, val = [], []
    for i in range(n, L - n):
        left = values[i - n:i]
        right = values[i + 1:i + n + 1]
        v = values[i]
        if kind == "high" and v > left.max() and v >= right.max():
            idx.append(i)
            val.append(v)
        elif kind == "low" and v < left.min() and v <= right.min():
            idx.append(i)
            val.append(v)
    return np.asarray(idx, dtype=int), np.asarray(val, dtype=float)


class SMCLiquiditySweep(Strategy):
    """Liquidity Sweep + Market Structure Shift + Displacement + Fibonacci + Fair Value Gap.

    A deterministic, causal port of the discretionary Smart Money Concepts setup. A single
    setup is tracked at a time through a five-stage state machine (bullish shown; short mirrors):

      1. Sweep        low pierces the most recent confirmed swing low and closes back above it.
      2. MSS          a later bar closes above the prior swing high (the "lower high").
      3. Displacement the sweep->break leg spans >= disp_atr_mult ATRs and the break candle has a
                      body/range >= disp_body_ratio (small wicks). Else the setup is discarded.
      4. Fibonacci    the 50%-61.8% retracement of the sweep-low -> displacement-high leg.
      5. FVG + entry  a bullish 3-candle fair-value gap inside the leg that overlaps the fib zone
                      defines the entry window; price must tap it and print a bullish confirmation
                      candle. Entry fills next open.

    Stop is structural (beyond the sweep, with an ATR buffer); take-profit is a fixed reward:risk.
    Sizing risks risk_pct of equity over the stop distance. Management: move the stop to breakeven
    after be_trigger_r R, and optionally exit early if structure breaks against the trade.

    Engine-fit notes: pivots are confirmed pivot_n bars after they form (no look-ahead). The
    breakeven move mutates the live position's stop_loss, which the engine honours on the *next*
    bar's bracket check. The structure exit uses ctx.close(), filling at the next open.
    """

    name = "Liquidity Sweep + MSS + FVG (SMC)"
    params = {
        "pivot_n": ("int", 2, 1, 20, "Swing pivot half-window (fractal strength)"),
        "pool_lookback": ("int", 50, 5, 500, "Bars back to find the swing (liquidity pool) to sweep"),
        "atr_len": ("int", 14, 2, 100, "ATR length (displacement sizing & stop buffer)"),
        "disp_atr_mult": ("float", 1.5, 0.1, 10.0, "Min displacement leg size, in ATRs"),
        "disp_body_ratio": ("float", 0.5, 0.0, 1.0, "Min body/range of the MSS break candle"),
        "fib_lo": ("float", 0.5, 0.0, 1.0, "Fib zone near bound (e.g. 0.50)"),
        "fib_hi": ("float", 0.618, 0.0, 1.0, "Fib zone far bound (e.g. 0.618)"),
        "setup_expiry": ("int", 20, 1, 500, "Max bars to wait between stages before reset"),
        "sl_buffer_atr": ("float", 0.25, 0.0, 5.0, "Stop buffer beyond the sweep, in ATRs"),
        "rr": ("float", 2.0, 0.5, 10.0, "Reward:risk for take-profit"),
        "risk_pct": ("float", 0.01, 0.001, 0.05, "Risk per trade (fraction of equity)"),
        "be_trigger_r": ("float", 1.0, 0.0, 10.0, "Move stop to breakeven after this many R (0 = off)"),
        "use_structure_exit": ("bool", True, None, None, "Exit early if structure breaks against the trade"),
    }

    # ------------------------------------------------------------------ lifecycle
    def on_start(self, ctx):
        df = ctx.data
        self._o = df["open"].to_numpy(float)
        self._h = df["high"].to_numpy(float)
        self._l = df["low"].to_numpy(float)
        self._c = df["close"].to_numpy(float)
        n = self.p["pivot_n"]
        self._ph_idx, self._ph_val = _pivots(self._h, n, "high")
        self._pl_idx, self._pl_val = _pivots(self._l, n, "low")
        self._atr = _wilder_atr(self._h, self._l, self._c, self.p["atr_len"])
        self._warmup = max(self.p["atr_len"] + 1, 2 * n + 2)
        self._reset()
        # management state for an open position
        self._mgmt_R = None
        self._be_done = False
        self._was_pos = False

    def _reset(self):
        self._state = "idle"
        self._dir = None
        self._sweep_px = None      # leg origin: swept low (long) / swept high (short)
        self._sweep_idx = None
        self._ref = None           # level whose break confirms the MSS
        self._disp_ext = None      # running displacement extreme
        self._expire = None
        self._zone = None          # (lo, hi) entry zone = fib ∩ FVG
        self._stop = None

    # ------------------------------------------------------------------ pivots
    def _last_pivot(self, idx_arr, val_arr, i, lookback=None):
        """Most recent pivot confirmed by bar i (pivot index <= i - pivot_n), optionally
        within `lookback` bars. Returns (index, price) or None."""
        if len(idx_arr) == 0:
            return None
        mask = idx_arr <= i - self.p["pivot_n"]
        if lookback is not None:
            mask &= idx_arr >= i - lookback
        cand = np.where(mask)[0]
        if len(cand) == 0:
            return None
        j = cand[-1]
        return int(idx_arr[j]), float(val_arr[j])

    # ------------------------------------------------------------------ main loop
    def on_bar(self, ctx):
        i = ctx.index
        if i < self._warmup:
            return

        pos = ctx.position
        if pos is not None:
            self._manage(ctx, pos, i)
            self._was_pos = True
            return
        if self._was_pos:                       # a trade just closed: clear management state
            self._mgmt_R, self._be_done, self._was_pos = None, False, False

        if self._state != "idle" and self._expire is not None and i > self._expire:
            self._reset()

        if self._state == "idle":
            self._try_sweep(i)
        elif self._state == "swept":
            self._process_swept(ctx, i)
        elif self._state == "await_entry":
            self._process_await(ctx, i)

    # ------------------------------------------------------------------ stage 1: sweep
    def _try_sweep(self, i):
        c = self._c[i]
        lb = self.p["pool_lookback"]
        pool_lo = self._last_pivot(self._pl_idx, self._pl_val, i, lb)
        if pool_lo is not None and self._l[i] < pool_lo[1] < c:
            ref = self._last_pivot(self._ph_idx, self._ph_val, i)   # the lower high to break
            if ref is not None:
                self._arm("long", self._l[i], i, ref[1])
                return
        pool_hi = self._last_pivot(self._ph_idx, self._ph_val, i, lb)
        if pool_hi is not None and self._h[i] > pool_hi[1] > c:
            ref = self._last_pivot(self._pl_idx, self._pl_val, i)
            if ref is not None:
                self._arm("short", self._h[i], i, ref[1])

    def _arm(self, direction, sweep_px, i, ref):
        self._state = "swept"
        self._dir = direction
        self._sweep_px = sweep_px
        self._sweep_idx = i
        self._ref = ref
        self._disp_ext = self._h[i] if direction == "long" else self._l[i]
        self._expire = i + self.p["setup_expiry"]

    # ------------------------------------------------------------------ stage 2: MSS
    def _process_swept(self, ctx, i):
        c = self._c[i]
        if self._dir == "long":
            self._disp_ext = max(self._disp_ext, self._h[i])
            if c < self._sweep_px:              # fell back through the sweep: setup failed
                self._reset()
            elif c > self._ref:                 # broke the lower high: MSS
                self._confirm_mss(i)
        else:
            self._disp_ext = min(self._disp_ext, self._l[i])
            if c > self._sweep_px:
                self._reset()
            elif c < self._ref:
                self._confirm_mss(i)

    # ------------------------------------------------------------ stage 3-5: displacement, fib, FVG
    def _confirm_mss(self, i):
        atr = self._atr[i]
        if np.isnan(atr) or atr <= 0:
            self._reset()
            return
        if self._dir == "long":
            H, L = self._disp_ext, self._sweep_px
        else:
            H, L = self._sweep_px, self._disp_ext
        leg = H - L
        if leg < self.p["disp_atr_mult"] * atr:          # weak displacement
            self._reset()
            return
        if self._body_ratio(i) < self.p["disp_body_ratio"]:
            self._reset()
            return

        zone_lo = H - self.p["fib_hi"] * leg
        zone_hi = H - self.p["fib_lo"] * leg             # zone_lo <= zone_hi
        fvg = self._find_fvg(self._sweep_idx, i, self._dir, zone_lo, zone_hi)
        if fvg is None:
            self._reset()
            return
        lo = max(zone_lo, fvg[0])
        hi = min(zone_hi, fvg[1])
        if lo >= hi:
            self._reset()
            return
        self._zone = (lo, hi)
        self._state = "await_entry"
        self._expire = i + self.p["setup_expiry"]

    def _body_ratio(self, i):
        rng = self._h[i] - self._l[i]
        return abs(self._c[i] - self._o[i]) / rng if rng > 0 else 0.0

    def _find_fvg(self, start, end, direction, zone_lo, zone_hi):
        """Most recent 3-candle fair-value gap inside bars [start, end] that overlaps the fib
        zone. Bullish gap = (high[m-1], low[m+1]) when low[m+1] > high[m-1]; bearish is the
        mirror. Returns (gap_lo, gap_hi) or None."""
        found = None
        for m in range(start + 1, end):
            if direction == "long":
                gap_lo, gap_hi = self._h[m - 1], self._l[m + 1]
            else:
                gap_lo, gap_hi = self._h[m + 1], self._l[m - 1]
            if gap_hi > gap_lo and gap_lo < zone_hi and gap_hi > zone_lo:
                found = (gap_lo, gap_hi)
        return found

    # ------------------------------------------------------------------ entry
    def _process_await(self, ctx, i):
        o, c = self._o[i], self._c[i]
        lo, hi = self._zone
        if self._dir == "long":
            if c < self._sweep_px:                       # invalidated
                self._reset()
                return
            if self._l[i] <= hi and c > o and c >= lo:   # tapped zone + bullish confirmation
                self._enter(ctx, i, "long")
        else:
            if c > self._sweep_px:
                self._reset()
                return
            if self._h[i] >= lo and c < o and c <= hi:
                self._enter(ctx, i, "short")

    def _enter(self, ctx, i, direction):
        c, atr = self._c[i], self._atr[i]
        if direction == "long":
            stop = self._sweep_px - self.p["sl_buffer_atr"] * atr
            R = c - stop
            tp = c + self.p["rr"] * R
        else:
            stop = self._sweep_px + self.p["sl_buffer_atr"] * atr
            R = stop - c
            tp = c - self.p["rr"] * R
        if R <= 0:
            self._reset()
            return
        size = ctx.size_for_risk(self.p["risk_pct"], c, stop)
        if size <= 0:
            self._reset()
            return
        ctx.enter(direction, size, stop_loss=stop, take_profit=tp, tag=f"smc_{direction}")
        self._mgmt_R, self._be_done = R, False
        self._reset()                                    # FSM idle; trade now managed in _manage

    # ------------------------------------------------------------------ trade management
    def _manage(self, ctx, pos, i):
        if self._mgmt_R is None and pos.stop_loss is not None:
            self._mgmt_R = abs(pos.entry_price - pos.stop_loss)
        R = self._mgmt_R

        # move stop to breakeven once price has run be_trigger_r R in favour
        if R and not self._be_done and self.p["be_trigger_r"] > 0:
            trig = self.p["be_trigger_r"] * R
            if pos.direction == "long" and self._h[i] >= pos.entry_price + trig:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price) if pos.stop_loss else pos.entry_price
                self._be_done = True
            elif pos.direction == "short" and self._l[i] <= pos.entry_price - trig:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price) if pos.stop_loss else pos.entry_price
                self._be_done = True

        # exit early if market structure breaks against the trade
        if self.p["use_structure_exit"]:
            c = self._c[i]
            if pos.direction == "long":
                piv = self._last_pivot(self._pl_idx, self._pl_val, i)
                if piv is not None and c < piv[1]:
                    ctx.close("structure_break")
            else:
                piv = self._last_pivot(self._ph_idx, self._ph_val, i)
                if piv is not None and c > piv[1]:
                    ctx.close("structure_break")
