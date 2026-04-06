"""Base abstract class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import pandas as pd


class BaseStrategy(ABC):
    """Base class for all trading strategies.

    Every strategy must implement the following abstract methods:
    - `on_data`: Process new data and optionally return a trade signal
    - `reset`: Reset strategy state for a new trading session
    - `get_state`: Return strategy state for persistence
    - `set_state`: Restore strategy state from persistence
    """

    name: str = "base"
    parameters: Dict[str, Any] = {}

    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with configuration."""
        ...

    @abstractmethod
    def on_data(self, row: pd.Series) -> Optional[dict]:
        """Process a single bar / tick and optionally return a signal.

        Returns:
            dict with keys at minimum:
                - signal_type: "entry" or "exit"
                - direction: "long" or "short" (for entries)
                - entry_price: float (for entries)
                - stop_loss: float (for entries)
                - take_profit: float (for entries)
            or None if no signal
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset strategy state for a new trading session."""
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """Return strategy state for persistence."""
        ...

    @abstractmethod
    def set_state(self, state: dict):
        """Restore strategy state from persistence."""
        ...

    def validate_signal(self, signal: dict) -> bool:
        """Validate signal before execution."""
        required_keys = {
            "signal_type",
            "direction",
            "entry_price",
            "stop_loss",
            "take_profit",
        }
        return required_keys.issubset(signal.keys())

    def check_exit(self, row: pd.Series) -> Optional[dict]:
        """Override in subclass if exit logic depends on realtime price.
        Default: no exit signal."""
        return None
