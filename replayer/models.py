from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class OrderReq:
    client_id: str
    side: str                 # "buy" | "sell"
    order_type: str           # "market" | "limit" | "stop"
    qty_lots: float
    price: Optional[float] = None      # required for limit/stop
    sl: Optional[float] = None
    tp: Optional[float] = None


@dataclass
class WorkingOrder:
    client_id: str
    side: str
    order_type: str
    qty_lots: float
    price: float
    sl: Optional[float]
    tp: Optional[float]

    def to_dict(self):
        return asdict(self)


@dataclass
class Fill:
    client_id: str
    side: str
    qty_lots: float
    fill_price: float
    ts: int


@dataclass
class BrokerEvent:
    """One state change emitted by the broker for logging + streaming."""
    type: str                 # "fill" | "position_close" | "order_reject" | "margin_stop" | ...
    ts: int
    data: dict = field(default_factory=dict)
