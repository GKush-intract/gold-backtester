from __future__ import annotations

from typing import Optional

from src.engine import (BacktestConfig, _buy_fill, _sell_fill, _resolve_bracket,
                        _trade_pnl, _unrealized)
from src.strategy import OZ_PER_LOT, Position, clamp_lots

from .models import BrokerEvent, OrderReq, WorkingOrder


class LiveBroker:
    """Incremental single-position broker. Call on_bar(bar) for each m1 bar to fill resting
    orders and resolve SL/TP; submit()/close_position() act at the current bar. Mirrors
    src/engine.py fill and cost semantics so recorded sessions re-simulate faithfully."""

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.balance = cfg.opening_balance
        self.equity = cfg.opening_balance
        self.position: Optional[Position] = None
        self.orders: list[WorkingOrder] = []
        self._bar: Optional[dict] = None
        self.stopped = False

    def _price(self) -> float:
        return self._bar["c"] if self._bar else 0.0

    def _dir(self, side: str) -> str:
        return "long" if side == "buy" else "short"

    def _open_position(self, side, qty_lots, base_price, sl, tp, ts) -> BrokerEvent:
        direction = self._dir(side)
        fill = _buy_fill(base_price, self.cfg) if side == "buy" else _sell_fill(base_price, self.cfg)
        size = clamp_lots(qty_lots, self.balance, fill, self.cfg.max_leverage)
        if size <= 0:
            return BrokerEvent("order_reject", ts, {"reason": "size_zero_or_leverage"})
        init_risk = (abs(fill - sl) * size * OZ_PER_LOT) if sl is not None else None
        self.position = Position(direction=direction, size=size, entry_price=fill,
                                 entry_time=ts, entry_index=0, entry_equity=self.balance,
                                 stop_loss=sl, take_profit=tp, initial_risk=init_risk,
                                 tag="", trail=None, trail_level=None)
        return BrokerEvent("fill", ts, {"fill_price": fill, "qty_lots": size,
                                        "side": side, "direction": direction,
                                        "sl": sl, "tp": tp})

    def _close(self, base_price, reason, ts) -> BrokerEvent:
        pos = self.position
        exit_fill = _sell_fill(base_price, self.cfg) if pos.direction == "long" else _buy_fill(base_price, self.cfg)
        pnl = _trade_pnl(pos, exit_fill, self.cfg)
        self.balance += pnl
        r = pnl / pos.initial_risk if pos.initial_risk else None
        self.position = None
        return BrokerEvent("position_close", ts,
                           {"exit_price": exit_fill, "reason": reason, "pnl": pnl,
                            "r_multiple": r, "direction": pos.direction})

    def submit(self, req: OrderReq) -> BrokerEvent:
        ts = self._bar["t"] if self._bar else 0
        if req.qty_lots <= 0:
            return BrokerEvent("order_reject", ts, {"client_id": req.client_id, "reason": "qty<=0"})
        if req.order_type == "market":
            if self.position is not None:
                return BrokerEvent("order_reject", ts,
                                   {"client_id": req.client_id, "reason": "position_open"})
            ev = self._open_position(req.side, req.qty_lots, self._price(), req.sl, req.tp, ts)
            ev.data["client_id"] = req.client_id
            return ev
        if req.price is None:
            return BrokerEvent("order_reject", ts, {"client_id": req.client_id, "reason": "no_price"})
        self.orders.append(WorkingOrder(req.client_id, req.side, req.order_type,
                                        req.qty_lots, req.price, req.sl, req.tp))
        return BrokerEvent("order_working", ts, {"client_id": req.client_id, "price": req.price,
                                                 "side": req.side, "order_type": req.order_type,
                                                 "qty_lots": req.qty_lots})

    def cancel(self, client_id: str) -> BrokerEvent:
        ts = self._bar["t"] if self._bar else 0
        self.orders = [o for o in self.orders if o.client_id != client_id]
        return BrokerEvent("order_cancel", ts, {"client_id": client_id})

    def close_position(self) -> Optional[BrokerEvent]:
        if self.position is None:
            return None
        return self._close(self._price(), "manual", self._bar["t"])

    def _order_triggers(self, o: WorkingOrder, h, l) -> bool:
        if o.order_type == "limit":
            return (o.side == "buy" and l <= o.price) or (o.side == "sell" and h >= o.price)
        return (o.side == "buy" and h >= o.price) or (o.side == "sell" and l <= o.price)

    def on_bar(self, bar: dict) -> list[BrokerEvent]:
        """Advance one m1 bar: resolve SL/TP intrabar, fill resting orders that trade through,
        mark-to-market. Returns the events produced."""
        self._bar = bar
        events: list[BrokerEvent] = []
        o, h, l, c, ts = bar["o"], bar["h"], bar["l"], bar["c"], bar["t"]

        if self.position is not None:
            hit = _resolve_bracket(self.position, o, h, l, self.cfg)
            if hit is not None:
                reason, base = hit
                events.append(self._close(base, "stop" if reason in ("stop", "trailing_stop") else "tp", ts))

        if self.position is None and self.orders:
            still: list[WorkingOrder] = []
            for ordr in self.orders:
                if self.position is None and self._order_triggers(ordr, h, l):
                    ev = self._open_position(ordr.side, ordr.qty_lots, ordr.price,
                                             ordr.sl, ordr.tp, ts)
                    ev.data["client_id"] = ordr.client_id
                    events.append(ev)
                else:
                    still.append(ordr)
            self.orders = still

        self.equity = self.balance + _unrealized(self.position, c)
        if self.equity <= 0 and not self.stopped:
            self.stopped = True
            if self.position is not None:
                events.append(self._close(c, "margin", ts))
            events.append(BrokerEvent("margin_stop", ts, {"equity": self.equity}))
        return events

    def account(self) -> dict:
        return {"balance": self.balance, "equity": self.equity,
                "position": self._position_view(), "orders": [o.to_dict() for o in self.orders]}

    def _position_view(self):
        p = self.position
        if p is None:
            return None
        return {"direction": p.direction, "size": p.size, "entry_price": p.entry_price,
                "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                "unrealized": _unrealized(p, self._price())}
