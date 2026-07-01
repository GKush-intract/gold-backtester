from replayer.matching import LiveBroker
from replayer.models import OrderReq
from src.engine import BacktestConfig


def bar(t, o, h, l, c, v=1.0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def fresh(**kw):
    kw.setdefault("max_leverage", 0.0)
    cfg = BacktestConfig(opening_balance=100_000.0, spread=0.0, slippage=0.0, **kw)
    return LiveBroker(cfg)


def test_market_order_fills_at_current_price():
    b = fresh()
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    ev = b.submit(OrderReq("o1", "buy", "market", 1.0))
    assert ev.type == "fill"
    assert ev.data["fill_price"] == 2000.0
    assert b.position is not None and b.position.direction == "long"


def test_buy_limit_rests_then_fills_on_touch():
    b = fresh()
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    ev = b.submit(OrderReq("o1", "buy", "limit", 1.0, price=1995.0))
    assert ev.type == "order_working"
    assert b.position is None
    evs = b.on_bar(bar(2, 1999, 2000, 1994, 1996))
    assert any(e.type == "fill" for e in evs)
    assert b.position.direction == "long"
    assert b.position.entry_price == 1995.0


def test_buy_stop_fills_when_high_crosses():
    b = fresh()
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    b.submit(OrderReq("o1", "buy", "stop", 1.0, price=2005.0))
    evs = b.on_bar(bar(2, 2001, 2006, 2000, 2004))
    assert any(e.type == "fill" for e in evs)
    assert b.position.entry_price == 2005.0


def test_stop_loss_resolves_intrabar():
    b = fresh()
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    b.submit(OrderReq("o1", "buy", "market", 1.0, sl=1990.0, tp=2020.0))
    evs = b.on_bar(bar(2, 1999, 2001, 1988, 1995))
    close = [e for e in evs if e.type == "position_close"]
    assert close and close[0].data["reason"] == "stop"
    assert b.position is None


def test_take_profit_resolves_intrabar():
    b = fresh()
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    b.submit(OrderReq("o1", "buy", "market", 1.0, sl=1990.0, tp=2020.0))
    evs = b.on_bar(bar(2, 2001, 2021, 2000, 2019))
    close = [e for e in evs if e.type == "position_close"]
    assert close and close[0].data["reason"] == "tp"


def test_leverage_cap_limits_size():
    b = fresh(max_leverage=1.0)
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    b.submit(OrderReq("o1", "buy", "market", 100.0))
    assert b.position.size <= 0.5 + 1e-9


def test_manual_close_and_equity():
    b = fresh()
    b.on_bar(bar(1, 2000, 2001, 1999, 2000))
    b.submit(OrderReq("o1", "buy", "market", 1.0))
    b.on_bar(bar(2, 2010, 2012, 2009, 2010))
    assert b.account()["equity"] > 100_000.0
    ev = b.close_position()
    assert ev.type == "position_close" and ev.data["reason"] == "manual"
    assert b.position is None
    assert b.account()["balance"] > 100_000.0
