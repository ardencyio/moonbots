"""yfinance data fetcher for futures and equities."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator

import pandas as pd

from backtest.data.fetcher import BaseDataFetcher


class YFinanceFetcher(BaseDataFetcher):
    """Fetch historical data from yfinance.

    Supports 1-minute bars for futures (NQ=F, ES=F) and equities.
    """

    name: str = "yfinance"
    supports_websocket: bool = False

    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1m",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data from yfinance."""
        import yfinance as yf

        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "1d": "1d",
        }
        interval = interval_map.get(resolution, "1m")

        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            prepost=True,
        )

        if df.empty:
            return df

        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            df = df.rename(
                columns={
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }
            )

        return df

    async def fetch_live_tick(self, symbol: str):
        """yfinance does not support real-time ticks."""
        raise NotImplementedError("yfinance does not provide real-time data")

    async def fetch_live_bars(self, symbol: str) -> AsyncGenerator[dict, None]:
        """yfinance does not support live bar streaming."""
        raise NotImplementedError("yfinance does not provide live bars")
        yield {}  # type: ignore[misc]

    def fetch(
        self, symbol: str, start: str, end: str, resolution: str = "1m"
    ) -> pd.DataFrame:
        """Synchronous fetch with parquet cache."""
        import asyncio
        from backtest.data.fetcher import get_cache_path, load_cache, save_cache

        cache_path = get_cache_path(symbol, start, end, resolution)
        cached = load_cache(cache_path)
        if cached is not None:
            return cached

        df = asyncio.run(
            self.fetch_historical(
                symbol,
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
                resolution,
            )
        )
        if not df.empty:
            save_cache(df, cache_path)
        return df
