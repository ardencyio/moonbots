"""Polygon.io data fetcher."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator

import pandas as pd

from backtest.data.fetcher import BaseDataFetcher


class PolygonFetcher(BaseDataFetcher):
    """Fetch data from Polygon.io REST API."""

    name: str = "polygon"
    supports_websocket: bool = True

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        from polygon import StocksClient

        if self._client is None:
            self._client = StocksClient(self.api_key)
        return self._client

    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1m",
    ) -> pd.DataFrame:
        aggs = self._get_client().get_aggs(
            symbol,
            1,
            resolution,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            adjusted=True,
        )

        rows = []
        for a in aggs:
            rows.append(
                {
                    "timestamp": pd.Timestamp(a.timestamp, unit="ms"),
                    "Open": a.open,
                    "High": a.high,
                    "Low": a.low,
                    "Close": a.close,
                    "Volume": a.volume,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df

    async def fetch_live_tick(self, symbol: str):
        """Fetch single latest tick from Polygon."""
        trades = self._get_client().get_trades(symbol, limit=1)
        return list(trades)

    async def fetch_live_bars(self, symbol: str) -> AsyncIterator[dict]:
        """Subscribe to aggregate minute bars via WebSocket."""
        from polygon import WebSocketClient

        async with WebSocketClient(self.api_key) as ws_client:
            ws_client.subscribe("AM." + symbol)
            async for data in ws_client:
                yield data
