"""Open Gate strategy for NQ/ES index futures.

Detects the first N-minute candle high/low at market open, waits for a
breakout, retest, and confirmation candle before entering.

Supports multiple timeframes: 1-minute, 5-minute, daily.
"""

from enum import Enum
from typing import Optional, Dict, Any

import pandas as pd

from backtest.strategies.base import BaseStrategy


class ConfirmationType(Enum):
    WICK_REJECTION = "wick_rejection"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"


class OpenGateStrategy(BaseStrategy):
    """Open Gate trading strategy."""

    name: str = "open_gate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gate_candle_minutes = self.config.get("gate_candle_minutes", 5)
        self.stop_buffer = self.config.get("stop_buffer_ticks", 1.0)
        self.atr_multiplier = self.config.get("atr_multiplier", 1.5)
        self.min_risk_reward = self.config.get("min_risk_reward", 2.0)
        self.session_open = self.config.get("session_open", "09:30:00")
        self.session_close = self.config.get("session_close", "16:00:00")
        self.use_market_hours = self.config.get("use_market_hours", True)

        # Internal state
        self.gate_high: Optional[float] = None
        self.gate_low: Optional[float] = None
        self.gate_set = False
        self.broke_above = False
        self.broke_below = False
        self.waiting_for_retest = False
        self.retest_direction: Optional[str] = None
        self.position_open = False
        self.position_direction: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None

    def reset(self):
        """Reset strategy state for a new trading session."""
        self.gate_high = None
        self.gate_low = None
        self.gate_set = False
        self.broke_above = False
        self.broke_below = False
        self.waiting_for_retest = False
        self.retest_direction = None
        self.position_open = False
        self.position_direction = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def get_state(self) -> dict:
        return {
            "gate_high": self.gate_high,
            "gate_low": self.gate_low,
            "gate_set": self.gate_set,
            "broke_above": self.broke_above,
            "broke_below": self.broke_below,
            "waiting_for_retest": self.waiting_for_retest,
            "retest_direction": self.retest_direction,
            "position_open": self.position_open,
            "position_direction": self.position_direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }

    def set_state(self, state: dict):
        self.gate_high = state.get("gate_high")
        self.gate_low = state.get("gate_low")
        self.gate_set = state.get("gate_set", False)
        self.broke_above = state.get("broke_above", False)
        self.broke_below = state.get("broke_below", False)
        self.waiting_for_retest = state.get("waiting_for_retest", False)
        self.retest_direction = state.get("retest_direction")
        self.position_open = state.get("position_open", False)
        self.position_direction = state.get("position_direction")
        self.entry_price = state.get("entry_price")
        self.stop_loss = state.get("stop_loss")
        self.take_profit = state.get("take_profit")

    def detect_gate(self, df: pd.DataFrame, session_date=None) -> pd.DataFrame:
        """Find the first N-min candle high/low for a session.

        If session_date is None, uses the first row's index to determine date.

        When use_market_hours is True, filters to session open/close times.
        When False, uses the first N candles of each day as the gate.
        """
        if session_date is None and len(df) > 0:
            session_date = df.index[0]
        if session_date is None:
            self.gate_set = False
            return df[:0]

        if self.use_market_hours:
            session_open = pd.Timestamp(self.session_open).time()
            gate_end = (
                pd.Timestamp(self.session_open) + pd.Timedelta(minutes=self.gate_candle_minutes)
            ).time()

            mask = (
                (df.index.date == session_date.date())
                & (df.index.time >= session_open)
                & (df.index.time < gate_end)
            )
            first_candles = df.loc[mask]
        else:
            # Use first N candles of the day regardless of time
            daily_groups = df.groupby(df.index.date)
            current_day_mask = df.index.date == session_date.date()
            current_day_df = df.loc[current_day_mask]

            # Get first N candles from this day
            gate_end_idx = min(self.gate_candle_minutes, len(current_day_df))
            first_candles = current_day_df.iloc[:gate_end_idx]

        if len(first_candles) > 0:
            self.gate_high = first_candles["High"].max()
            self.gate_low = first_candles["Low"].min()
            self.gate_set = True
        return first_candles

    def detect_breakout(self, row: pd.Series) -> None:
        """Detect if price broke through the gate."""
        if not self.gate_set or self.waiting_for_retest:
            return

        if row["Close"] > self.gate_high and not self.broke_above:
            self.broke_above = True
            self.waiting_for_retest = True
            self.retest_direction = "bullish"
        elif row["Close"] < self.gate_low and not self.broke_below:
            self.broke_below = True
            self.waiting_for_retest = True
            self.retest_direction = "bearish"

    def detect_retest(self, row: pd.Series) -> bool:
        """Detect if price returned to test the broken level."""
        if not self.waiting_for_retest:
            return False

        if self.retest_direction == "bullish" and row["Low"] <= self.gate_high:
            return True
        elif self.retest_direction == "bearish" and row["High"] >= self.gate_low:
            return True
        return False

    def check_confirmation(
        self, df: pd.DataFrame, idx: int
    ) -> Dict[str, Any]:
        """Check for entry confirmation signals.

        Looks for wick rejection or engulfing patterns.
        """
        if idx < 2:
            return {"confirmed": False, "type": None}

        current = df.iloc[idx]
        prev1 = df.iloc[idx - 1]

        if self.retest_direction == "bullish":
            # Bullish wick rejection
            if current["Low"] < prev1["Low"] and current["Close"] > current["Open"]:
                return {"confirmed": True, "type": ConfirmationType.WICK_REJECTION}
            # Bullish engulfing
            if (
                prev1["Close"] < prev1["Open"]
                and current["Close"] > prev1["Open"]
                and current["Open"] < prev1["Close"]
            ):
                return {"confirmed": True, "type": ConfirmationType.BULLISH_ENGULFING}

        elif self.retest_direction == "bearish":
            # Bearish wick rejection
            if current["High"] > prev1["High"] and current["Close"] < current["Open"]:
                return {"confirmed": True, "type": ConfirmationType.WICK_REJECTION}
            # Bearish engulfing
            if (
                prev1["Close"] > prev1["Open"]
                and current["Close"] < prev1["Open"]
                and current["Open"] > prev1["Close"]
            ):
                return {"confirmed": True, "type": ConfirmationType.BEARISH_ENGULFING}

        return {"confirmed": False, "type": None}

    def calculate_stop_loss(self, direction: str) -> float:
        """Calculate stop loss based on gate levels.

        Long: stop below gate_low
        Short: stop above gate_high
        """
        buffer = self.stop_buffer
        if direction == "long":
            return (self.gate_low or 0) - buffer
        return (self.gate_high or 0) + buffer

    def calculate_take_profit(self, entry: float, stop: float) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(entry - stop)
        if risk == 0:
            return entry
        rr = self.min_risk_reward
        if self.position_direction == "long":
            return entry + (risk * rr)
        return entry - (risk * rr)

    def check_exit(self, row: pd.Series) -> Optional[dict]:
        """Public method to check for exit signals."""
        return self._check_position_exit(row)

    def on_data(self, row: pd.Series, all_data: pd.DataFrame = None) -> Optional[dict]:
        """Process a single bar. Returns signal dict or None.

        Args:
            row: Current bar data
            all_data: Full dataset for gate detection (required if gate not set)

        Returns:
            Signal dict with 'signal_type', 'direction', etc. or None
        """
        # If we have an open position, check for exits
        if self.position_open:
            return self._check_position_exit(row)

        # If gate is not set, try to detect it from data
        if not self.gate_set and all_data is not None:
            self.detect_gate(all_data)

        # If gate is still not set, we can't trade yet
        if not self.gate_set:
            return None

        # Detect breakout
        if not self.waiting_for_retest:
            self.detect_breakout(row)

        # Detect retest
        if self.waiting_for_retest and self.detect_retest(row):
            # Store entry information - direction must be set BEFORE calculate_take_profit
            if self.retest_direction == "bullish":
                self.position_direction = "long"
                self.entry_price = row["Close"]
                self.stop_loss = self.calculate_stop_loss("long")
                self.take_profit = self.calculate_take_profit(self.entry_price, self.stop_loss)
                self.position_open = True
                return {
                    "signal_type": "entry",
                    "direction": "long",
                    "entry_price": self.entry_price,
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                }
            elif self.retest_direction == "bearish":
                self.position_direction = "short"
                self.entry_price = row["Close"]
                self.stop_loss = self.calculate_stop_loss("short")
                self.take_profit = self.calculate_take_profit(self.entry_price, self.stop_loss)
                self.position_open = True
                return {
                    "signal_type": "entry",
                    "direction": "short",
                    "entry_price": self.entry_price,
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                }

        return None

    def _check_position_exit(self, row: pd.Series) -> Optional[dict]:
        """Check stop loss / take profit for open position."""
        if not self.position_open or self.stop_loss is None:
            return None

        if self.position_direction == "long":
            if row["Low"] <= self.stop_loss:
                return self._close_position(
                    price=self.stop_loss, reason="stop_loss"
                )
            if self.take_profit and row["High"] >= self.take_profit:
                return self._close_position(
                    price=self.take_profit, reason="take_profit"
                )
        else:  # short
            if self.stop_loss and row["High"] >= self.stop_loss:
                return self._close_position(
                    price=self.stop_loss, reason="stop_loss"
                )
            if self.take_profit and row["Low"] <= self.take_profit:
                return self._close_position(
                    price=self.take_profit, reason="take_profit"
                )
        return None

    def _close_position(self, price: float, reason: str) -> dict:
        """Close the current position and return exit signal."""
        pnl = 0
        if self.position_direction == "long":
            pnl = price - (self.entry_price or 0)
        else:
            pnl = (self.entry_price or 0) - price

        result = {
            "signal_type": "exit",
            "direction": self.position_direction,
            "exit_price": price,
            "reason": reason,
            "pnl": pnl,
        }

        # Reset position
        self.position_open = False
        self.position_direction = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

        # Reset breakout state for next trade
        self.waiting_for_retest = False
        self.broke_above = False
        self.broke_below = False
        self.retest_direction = None

        return result
