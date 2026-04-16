"""Polygon.io data fetcher."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator

import pandas as pd

from backtest.data.fetcher import BaseDataFetcher

_RESOLUTION_MAP: dict[str, tuple[str, int]] = {
    "1m": ("minute", 1),
    "5m": ("minute", 5),
    "15m": ("minute", 15),
    "1h": ("hour", 1),
    "1d": ("day", 1),
    "minute": ("minute", 1),
    "hour": ("hour", 1),
    "day": ("day", 1),
}


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

    def fetch(
        self, symbol: str, start: str, end: str, resolution: str = "1d"
    ) -> pd.DataFrame:
        """Synchronous fetch of historical OHLCV bars with parquet cache."""
        from backtest.data.fetcher import get_cache_path, load_cache, save_cache

        cache_path = get_cache_path(symbol, start, end, resolution)
        cached = load_cache(cache_path)
        if cached is not None:
            return cached

        timespan, multiplier = _RESOLUTION_MAP.get(resolution, ("day", 1))
        bars = self._get_client().get_aggregate_bars(
            symbol,
            start,
            end,
            timespan=timespan,
            multiplier=multiplier,
            full_range=True,
            run_parallel=False,
            adjusted=True,
            warnings=False,
            info=False,
        )
        if not bars:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": pd.Timestamp(b["t"], unit="ms"),
                "Open": b["o"],
                "High": b["h"],
                "Low": b["l"],
                "Close": b["c"],
                "Volume": b["v"],
            }
            for b in bars
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index)
        save_cache(df, cache_path)
        return df

    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1d",
    ) -> pd.DataFrame:
        return self.fetch(
            symbol,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            resolution,
        )

    async def fetch_live_tick(self, symbol: str):
        trades = self._get_client().get_trades(symbol, limit=1)
        return list(trades)

    async def fetch_live_bars(self, symbol: str) -> AsyncGenerator[dict, None]:
        raise NotImplementedError(
            "Live bar streaming requires polygon WebSocket subscription"
        )
        # Silence unreachable-yield — AsyncIterator protocol requires a yield
        yield {}  # type: ignore[misc]
