"""Data fetchers for market data."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator

import pandas as pd


class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers.

    Subclasses must implement fetch_historical and optionally
    fetch_live_tick / fetch_live_bars for real-time data.
    """

    name: str = "base"
    supports_websocket: bool = False

    @abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1m",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        ...

    @abstractmethod
    async def fetch_live_tick(self, symbol: str):
        """Fetch live tick data (websocket or polling)."""
        ...

    @abstractmethod
    async def fetch_live_bars(self, symbol: str) -> AsyncIterator[dict]:
        """Yield live bar updates."""
        ...

    def normalize(self, raw_data: dict, symbol: str) -> pd.DataFrame:
        """Normalize raw data to standard OHLCV format.

        Returns a DataFrame with columns: Open, High, Low, Close, Volume
        and a datetime index.
        """
        df = pd.DataFrame(raw_data)
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df.pop("timestamp"))
        required = {"Open", "High", "Low", "Close", "Volume"}
        if required.issubset(df.columns):
            return df
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
