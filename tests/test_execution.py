"""Tests for ExecutionHandler (synchronous)."""

import asyncio

import pytest

from shared.core.execution import (
    Order,
    OrderType,
    OrderSide,
    PaperExecutionHandler,
)


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_submit_order():
    async def _test():
        handler = PaperExecutionHandler()
        await handler.connect()
        order = Order(
            symbol="NQ=F",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
        )
        result = await handler.submit_order(order)
        assert result.status == "filled"
        assert result.filled_quantity == 1

    run_async(_test())


def test_unconnected_rejects():
    async def _test():
        handler = PaperExecutionHandler()
        order = Order(
            symbol="NQ=F",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1,
        )
        with pytest.raises(RuntimeError, match="Not connected"):
            await handler.submit_order(order)

    run_async(_test())


def test_get_open_orders():
    async def _test():
        handler = PaperExecutionHandler()
        await handler.connect()
        orders = await handler.get_open_orders()
        assert orders == []

    run_async(_test())
