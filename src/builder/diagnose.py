"""Token-efficient backtest digest for opt-in AI diagnosis.

build_digest() compresses a backtest (result + metrics + price context) into a
compact fixed-width text block (~1-2k tokens) that the revision interview can
reason about — sent to the API ONLY when the user explicitly clicks the
"Diagnose with AI" button, never automatically.
"""
from __future__ import annotations

import pandas as pd


def _fmt_trades(rows: pd.DataFrame) -> str:
    out = []
    for r in rows.itertuples():
        rr = f"{r.r_multiple:+.2f}R" if r.r_multiple == r.r_multiple else "  ?R"
        out.append(f"  {pd.Timestamp(r.entry_time):%m-%d %H:%M} {r.direction:<5} "
                   f"in {r.entry_price:.2f} out {r.exit_price:.2f} "
                   f"{r.pnl:+8.0f}$ {rr} {r.bars_held:>4}bars {r.exit_reason}")
    return "\n".join(out)


def build_digest(res, m: dict, data: pd.DataFrame) -> str:
    """Compact text digest of a backtest + price context for AI diagnosis."""
    tr = res.trades
    cfg = res.config
    lines = []
    lines.append(f"PERIOD {res.data_start:%Y-%m-%d} .. {res.data_end:%Y-%m-%d} | "
                 f"{len(data)} bars @ ~{res.timeframe_seconds/60:.0f}min | "
                 f"costs: spread {cfg.spread}, slip {cfg.slippage}, "
                 f"comm {cfg.commission_per_trade}$/trade + {cfg.commission_per_lot}$/lot | "
                 f"lev cap {cfg.max_leverage}x")
    lines.append(f"RESULT return {m['total_return_pct']*100:+.1f}% | PF {m['profit_factor']:.2f} | "
                 f"win {m['win_rate']*100:.1f}% | expectancy {m['expectancy_r']:.2f}R | "
                 f"maxDD {m['max_drawdown_pct']*100:.1f}% | trades {m['num_trades']}"
                 + (" | STOPPED OUT (equity hit 0)" if res.stopped_out else ""))

    if tr.empty:
        lines.append("NO TRADES — entry condition never fired on this data.")
        return "\n".join(lines)

    rm = tr["r_multiple"].dropna()
    if len(rm):
        buckets = pd.cut(rm, [-99, -2, -1, 0, 1, 2, 99],
                         labels=["<=-2R", "-2..-1R", "-1..0R", "0..1R", "1..2R", ">2R"])
        dist = buckets.value_counts().reindex(buckets.cat.categories, fill_value=0)
        lines.append("R DIST  " + "  ".join(f"{k}:{v}" for k, v in dist.items()))

    lines.append("EXIT REASONS (count / total pnl / avg R):")
    g = tr.groupby("exit_reason").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                      avg_r=("r_multiple", "mean"))
    for reason, row in g.sort_values("pnl").iterrows():
        ar = f"{row['avg_r']:+.2f}R" if row["avg_r"] == row["avg_r"] else "?R"
        lines.append(f"  {reason:<24} {int(row['n']):>5}  {row['pnl']:+10.0f}$  {ar}")

    et = pd.DatetimeIndex(tr["entry_time"])
    by_sess = tr.assign(sess=et.hour // 4 * 4).groupby("sess").agg(
        n=("pnl", "size"), pnl=("pnl", "sum"),
        win=("pnl", lambda s: (s > 0).mean()))
    lines.append("BY UTC SESSION (h / n / pnl / win%):")
    lines.append("  " + " | ".join(
        f"{int(s):02d}-{int(s)+4:02d}h n={int(r['n'])} {r['pnl']:+.0f}$ {r['win']*100:.0f}%"
        for s, r in by_sess.iterrows()))
    by_dow = tr.assign(d=et.day_name().str[:3]).groupby("d").agg(
        n=("pnl", "size"), pnl=("pnl", "sum"))
    lines.append("BY WEEKDAY: " + " | ".join(
        f"{d} n={int(r['n'])} {r['pnl']:+.0f}$" for d, r in by_dow.iterrows()))

    losses = (tr["pnl"] < 0).astype(int)
    streak = int((losses.groupby((losses != losses.shift()).cumsum()).cumsum()).max())
    lines.append(f"AVG bars held {tr['bars_held'].mean():.0f} | avg size {tr['size'].mean():.2f} lots | "
                 f"max consecutive losses {streak}")

    lines.append("5 WORST TRADES:")
    lines.append(_fmt_trades(tr.nsmallest(5, "pnl")))
    lines.append("5 BEST TRADES:")
    lines.append(_fmt_trades(tr.nlargest(5, "pnl")))

    # price context + strategy pnl per month — regime vs performance
    mo = data.resample("MS").agg(open=("open", "first"), high=("high", "max"),
                                 low=("low", "min"), close=("close", "last"))
    pnl_mo = tr.set_index(pd.DatetimeIndex(tr["entry_time"])).resample("MS")["pnl"].sum()
    lines.append("MONTHLY price context vs strategy pnl:")
    for ts, row in mo.iterrows():
        chg = (row["close"] / row["open"] - 1) * 100
        rng = (row["high"] - row["low"]) / row["open"] * 100
        p = pnl_mo.get(ts, 0.0)
        lines.append(f"  {ts:%Y-%m} close {row['close']:7.0f} chg {chg:+5.1f}% range {rng:4.1f}% "
                     f"| strat {p:+8.0f}$")

    eq = res.equity_curve
    dd_at = eq["drawdown_pct"].idxmin()
    lines.append(f"EQUITY {eq['equity'].iloc[0]:.0f} -> {eq['equity'].iloc[-1]:.0f}$; "
                 f"deepest drawdown at {dd_at:%Y-%m-%d %H:%M}")
    return "\n".join(lines)


DIAG_REQUEST = """\
The user clicked "Diagnose with AI". Below is a compact digest of the latest backtest of the
CURRENT strategy code, plus price context. Poke holes: identify the top reasons this strategy
underperforms — be specific (which exit reasons bleed money, which sessions/regimes hurt, cost
drag vs edge, stop placement, target asymmetry, trade frequency). Then propose the 2-3
highest-leverage concrete changes: parameter retunes first, logic changes second. Discuss with
the user and call finalize_revision only once a change is agreed.

=== BACKTEST DIGEST ===
"""
