import pandas as pd
import pytest

from replayer.market import MarketFeed


def make_feed(times_min, price=2000.0):
    """times_min: list of minute offsets from a base; builds a tiny MarketFeed."""
    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    rows = []
    for k in times_min:
        t = base + pd.Timedelta(minutes=k)
        rows.append({"timestamp": t, "open": price, "high": price + 1,
                     "low": price - 1, "close": price + 0.5, "volume": 1.0})
    df = pd.DataFrame(rows)
    return MarketFeed(df)


def test_iteration_in_order_and_end():
    feed = make_feed([0, 1, 2])
    seen = []
    while not feed.at_end:
        seen.append(feed.next_bar()["t"])
    assert seen == sorted(seen)
    assert len(seen) == 3
    assert feed.at_end


def test_candles_shape():
    feed = make_feed([0, 1])
    c = feed.candles()
    assert c[0].keys() == {"t", "o", "h", "l", "c", "v"}
    assert c[0]["t"] < c[1]["t"]


def test_seek_positions_cursor():
    feed = make_feed([0, 1, 2, 3])
    target = feed.candles()[2]["t"]
    feed.seek(target)
    assert feed.next_bar()["t"] == target


def test_skip_gap_advances_over_large_gap():
    # bars at 0, 1, then a 3000-minute jump (weekend-like) to 3001
    feed = make_feed([0, 1, 3001])
    feed.next_bar()  # consume bar 0
    feed.next_bar()  # consume bar 1; cursor now before the big gap
    jumped = feed.skip_gap(threshold_ms=60 * 60 * 1000)  # 1h threshold
    assert jumped is True
    assert feed.peek()["t"] == feed.candles()[2]["t"]
