from __future__ import annotations

from src.engine import BacktestConfig

from .market import MarketFeed
from .matching import LiveBroker
from .models import OrderReq
from .session import Session

SKIP_GAP_MS = 60 * 60 * 1000   # jump gaps larger than 1h (weekends / market close)


class ReplaySession:
    """Authoritative replay state for one WS connection: a feed cursor, a broker, and the
    session log. Pure logic — the transport (WebSocket) calls handle()/tick() and forwards the
    returned outbound messages. Every inbound action and outbound state change is logged."""

    def __init__(self, session: Session, feed: MarketFeed, cfg: BacktestConfig):
        self.session = session
        self.feed = feed
        self.broker = LiveBroker(cfg)
        self.playing = False
        self.speed = 1

    def _account_msg(self) -> dict:
        return {"type": "account", **self.broker.account()}

    def _bar_msg(self, bar: dict, broker_events: list) -> dict:
        return {"type": "bar", "bar": bar,
                "events": [{"type": e.type, "ts": e.ts, "data": e.data} for e in broker_events]}

    def tick(self) -> list[dict]:
        """Advance one bar. Returns outbound messages; logs bar-driven broker events."""
        if self.feed.at_end:
            self.playing = False
            self.session.log("clock", {"action": "end"}, market_ts=None)
            return [{"type": "clock_state", "playing": False, "at_end": True}]
        bar = self.feed.next_bar()
        events = self.broker.on_bar(bar)
        for e in events:
            self.session.log(e.type, e.data, market_ts=e.ts)
        if self.broker.stopped:
            self.playing = False
        return [self._bar_msg(bar, events), self._account_msg()]

    def handle(self, msg: dict) -> list[dict]:
        kind = msg.get("kind")
        peek = self.feed.peek()
        market_ts = peek["t"] if peek else None
        if kind == "control":
            return self._control(msg, market_ts)
        if kind == "order":
            return self._order(msg, market_ts)
        if kind == "cancel":
            ev = self.broker.cancel(msg["client_id"])
            self.session.log(ev.type, ev.data, market_ts=market_ts)
            return [self._account_msg()]
        if kind == "close_position":
            ev = self.broker.close_position()
            if ev:
                self.session.log(ev.type, ev.data, market_ts=market_ts)
            return [self._account_msg()]
        if kind in ("draw", "note_text", "tf_change", "indicator"):
            self.session.log(kind, msg.get("data", {}), market_ts=market_ts)
            return []
        return []

    def _control(self, msg, market_ts) -> list[dict]:
        action = msg["action"]
        if action == "play":
            self.playing = True
            self.speed = int(msg.get("speed", 1))
            self.session.log("clock", {"action": "play", "speed": self.speed}, market_ts=market_ts)
            return [{"type": "clock_state", "playing": True, "speed": self.speed}]
        if action == "pause":
            self.playing = False
            self.session.log("clock", {"action": "pause"}, market_ts=market_ts)
            return [{"type": "clock_state", "playing": False}]
        if action == "speed":
            self.speed = int(msg["speed"])
            self.session.log("clock", {"action": "speed", "speed": self.speed}, market_ts=market_ts)
            return [{"type": "clock_state", "speed": self.speed}]
        if action == "step":
            self.session.log("clock", {"action": "step"}, market_ts=market_ts)
            return self.tick()
        if action == "seek":
            # seek is chart/clock navigation only: it does NOT rewind an open position or
            # working orders (acceptable for step 1). It only re-marks the broker price.
            self.feed.seek(int(msg["to"]))
            peek = self.feed.peek()
            if peek is not None:
                self.broker.mark(peek)
            self.session.log("clock", {"action": "seek", "to": int(msg["to"])}, market_ts=int(msg["to"]))
            return [{"type": "seeked", "to": int(msg["to"])}, self._account_msg()]
        if action == "skip_gap":
            if self.feed.skip_gap(SKIP_GAP_MS):
                target = self.feed.peek()["t"]
                self.session.log("clock", {"action": "skip_gap", "to": target}, market_ts=target)
                return [{"type": "seeked", "to": target}]
            return []
        return []

    def _order(self, msg, market_ts) -> list[dict]:
        d = msg["data"]
        req = OrderReq(client_id=d["client_id"], side=d["side"], order_type=d["order_type"],
                       qty_lots=float(d["qty_lots"]), price=d.get("price"),
                       sl=d.get("sl"), tp=d.get("tp"))
        self.session.log("order_submit", d, market_ts=market_ts)
        ev = self.broker.submit(req)
        self.session.log(ev.type, ev.data, market_ts=market_ts)
        return [{"type": ev.type, "data": ev.data}, self._account_msg()]
