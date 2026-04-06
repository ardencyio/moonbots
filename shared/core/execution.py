"""Execution handler abstraction for brokers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    status: str = "pending"
    filled_price: Optional[float] = None
    filled_quantity: float = 0
    created_at: datetime = field(default_factory=datetime.now)


class ExecutionHandler(ABC):
    """Abstract base for execution handlers."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        ...

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    async def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        ...

    @abstractmethod
    async def get_account_balance(self) -> dict:
        """Get account balance and buying power."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Get current positions."""
        ...

    async def close_position(self, symbol: str, quantity: float) -> Order:
        """Close a position."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        return await self.submit_order(order)


class PaperExecutionHandler(ExecutionHandler):
    """Paper trading execution handler — simulates fills at close price."""

    def __init__(self):
        self.orders: list[Order] = []
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def submit_order(self, order: Order) -> Order:
        if not self.connected:
            raise RuntimeError("Not connected")
        # Simulate immediate fill at market
        order.status = "filled"
        order.filled_price = order.limit_price or 0
        order.filled_quantity = order.quantity
        self.orders.append(order)
        return order

    async def cancel_order(self, order_id: str) -> bool:
        for o in self.orders:
            if o.order_id == order_id and o.status == "pending":
                o.status = "cancelled"
                return True
        return False

    async def get_open_orders(self) -> list[Order]:
        return [o for o in self.orders if o.status == "pending"]

    async def get_account_balance(self) -> dict:
        return {"cash": 0, "buying_power": 0}

    async def get_positions(self) -> list[dict]:
        return []
