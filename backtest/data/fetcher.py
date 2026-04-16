"""Data fetchers for market data."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import pandas as pd

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


def get_cache_path(
    symbol: str,
    start: str,
    end: str,
    resolution: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> Path:
    safe = symbol.replace("/", "-").replace(":", "-").replace("=", "-")
    return cache_dir / f"{safe}_{start}_{end}_{resolution}.parquet"


def load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


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
    async def fetch_live_bars(self, symbol: str) -> AsyncGenerator[dict, None]:
        """Yield live bar updates."""
        yield {}  # pragma: no cover

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
