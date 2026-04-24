"""Open Gate MNQ strategy — 1:1 port of scripts/pine/open_gate.pine.

Full feature set including position sizing (contracts), RM 1 trailing stop,
RM 2 break-even stop, and RM 3 time exit. Do not modify open_gate.py.
"""

from datetime import time
from typing import Optional, Dict, Any

import pandas as pd

from backtest.strategies.base import BaseStrategy


class OpenGateMNQStrategy(BaseStrategy):
    """Open Gate strategy with full Pine Script parity for MNQ futures."""

    name: str = "open_gate_mnq"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Position sizing
        self.contracts = int(self.config.get("contracts", 1))

        # Gate settings
        self.gate_candle_minutes = self.config.get("gate_candle_minutes", 1)
        self.stop_buffer_ticks = self.config.get("stop_buffer_ticks", 1.0)
        self.min_risk_reward = self.config.get("min_risk_reward", 2.0)
        self.session_window = self.config.get("session_window", "0930-1600")
        self.use_market_hours = self.config.get("use_market_hours", True)
        self.use_confirmation = self.config.get("use_confirmation", True)

        # RM 1 — Trailing stop
        self.use_trailing = self.config.get("use_trailing", False)
        self.trail_offset_pts = self.config.get("trail_offset_pts", 15.0)

        # RM 2 — Break-even
        self.use_break_even = self.config.get("use_break_even", False)
        self.be_trigger_pts = self.config.get("be_trigger_pts", 20.0)
        self.be_buffer_pts = self.config.get("be_buffer_pts", 2.0)

        # RM 3 — Time exit
        self.use_time_exit = self.config.get("use_time_exit", False)
        self.exit_hour = self.config.get("exit_hour", 15)
        self.exit_minute = self.config.get("exit_minute", 45)

        # Parse session window "0930-1600" → (time(9,30), time(16,0))
        self._session_start, self._session_end = self._parse_session_window(
            self.session_window
        )

        # Per-day state
        self.gate_high: Optional[float] = None
        self.gate_low: Optional[float] = None
        self.gate_set = False
        self.gate_bar_count = 0
        self.broke_above = False
        self.broke_below = False
        self.waiting_for_retest = False
        self.retest_direction: Optional[str] = None
        self.be_done = False

        # Trade state (persists across day boundary while position is open)
        self.position_open = False
        self.position_direction: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.active_stop: Optional[float] = None
        self.active_tp: Optional[float] = None
        self.trail_extreme: Optional[float] = None

        # Track last-seen date for new-day detection
        self._last_date = None

    @staticmethod
    def _parse_session_window(window: str):
        """Parse 'HHMM-HHMM' into two time objects."""
        start_str, end_str = window.split("-")
        start = time(int(start_str[:2]), int(start_str[2:]))
        end = time(int(end_str[:2]), int(end_str[2:]))
        return start, end

    def reset(self):
        """Reset all strategy state."""
        self.gate_high = None
        self.gate_low = None
        self.gate_set = False
        self.gate_bar_count = 0
        self.broke_above = False
        self.broke_below = False
        self.waiting_for_retest = False
        self.retest_direction = None
        self.be_done = False
        self.position_open = False
        self.position_direction = None
        self.entry_price = None
        self.active_stop = None
        self.active_tp = None
        self.trail_extreme = None
        self._last_date = None

    def _reset_day(self):
        """Reset per-day gate and breakout state. Mirrors Pine's new_day block."""
        self.gate_high = None
        self.gate_low = None
        self.gate_set = False
        self.gate_bar_count = 0
        self.broke_above = False
        self.broke_below = False
        self.waiting_for_retest = False
        self.retest_direction = None
        self.be_done = False

    def _reset_trade(self):
        """Re-arm for the next trade after a position closes. Mirrors Pine's flat+was_in_trade block."""
        self.waiting_for_retest = False
        self.broke_above = False
        self.broke_below = False
        self.retest_direction = None
        self.active_stop = None
        self.active_tp = None
        self.be_done = False

    def get_state(self) -> dict:
        return {
            "gate_high": self.gate_high,
            "gate_low": self.gate_low,
            "gate_set": self.gate_set,
            "gate_bar_count": self.gate_bar_count,
            "broke_above": self.broke_above,
            "broke_below": self.broke_below,
            "waiting_for_retest": self.waiting_for_retest,
            "retest_direction": self.retest_direction,
            "be_done": self.be_done,
            "position_open": self.position_open,
            "position_direction": self.position_direction,
            "entry_price": self.entry_price,
            "active_stop": self.active_stop,
            "active_tp": self.active_tp,
            "trail_extreme": self.trail_extreme,
        }

    def set_state(self, state: dict):
        self.gate_high = state.get("gate_high")
        self.gate_low = state.get("gate_low")
        self.gate_set = state.get("gate_set", False)
        self.gate_bar_count = state.get("gate_bar_count", 0)
        self.broke_above = state.get("broke_above", False)
        self.broke_below = state.get("broke_below", False)
        self.waiting_for_retest = state.get("waiting_for_retest", False)
        self.retest_direction = state.get("retest_direction")
        self.be_done = state.get("be_done", False)
        self.position_open = state.get("position_open", False)
        self.position_direction = state.get("position_direction")
        self.entry_price = state.get("entry_price")
        self.active_stop = state.get("active_stop")
        self.active_tp = state.get("active_tp")
        self.trail_extreme = state.get("trail_extreme")

    def detect_gate(self, row: pd.Series) -> None:
        """Update gate high/low from a single bar. Called bar-by-bar in on_data."""
        bar_time = row.name.time()
        in_full_session = self._session_start <= bar_time < self._session_end

        if self.use_market_hours:
            in_gate_window = in_full_session and self.gate_bar_count < self.gate_candle_minutes
        else:
            in_gate_window = self.gate_bar_count < self.gate_candle_minutes

        if in_gate_window:
            high = row["High"]
            low = row["Low"]
            self.gate_high = high if self.gate_high is None else max(self.gate_high, high)
            self.gate_low = low if self.gate_low is None else min(self.gate_low, low)
            self.gate_bar_count += 1
            if self.gate_bar_count >= self.gate_candle_minutes:
                self.gate_set = True

    def detect_breakout(self, row: pd.Series) -> None:
        """Detect close through gate_high or gate_low."""
        if not self.gate_set or self.waiting_for_retest:
            return

        bar_time = row.name.time()
        in_full_session = self._session_start <= bar_time < self._session_end
        if not in_full_session:
            return

        close = row["Close"]
        if close > self.gate_high and not self.broke_above:
            self.broke_above = True
            self.waiting_for_retest = True
            self.retest_direction = "bullish"
        elif close < self.gate_low and not self.broke_below:
            self.broke_below = True
            self.waiting_for_retest = True
            self.retest_direction = "bearish"

    def detect_retest(self, row: pd.Series) -> bool:
        """Return True if price returned to test the broken gate level."""
        if not self.waiting_for_retest:
            return False
        if self.retest_direction == "bullish" and row["Low"] <= self.gate_high:
            return True
        if self.retest_direction == "bearish" and row["High"] >= self.gate_low:
            return True
        return False

    def check_confirmation(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Check for wick-rejection or engulfing confirmation. Mirrors Pine exactly."""
        if idx < 1:
            return {"confirmed": False}

        current = df.iloc[idx]
        prev = df.iloc[idx - 1]

        if self.retest_direction == "bullish":
            bullish_wick_rejection = current["Low"] < prev["Low"] and current["Close"] > current["Open"]
            bullish_engulfing = (
                prev["Close"] < prev["Open"]
                and current["Close"] > prev["Open"]
                and current["Open"] < prev["Close"]
            )
            if bullish_wick_rejection or bullish_engulfing:
                return {"confirmed": True}

        elif self.retest_direction == "bearish":
            bearish_wick_rejection = current["High"] > prev["High"] and current["Close"] < current["Open"]
            bearish_engulfing = (
                prev["Close"] > prev["Open"]
                and current["Close"] < prev["Open"]
                and current["Open"] > prev["Close"]
            )
            if bearish_wick_rejection or bearish_engulfing:
                return {"confirmed": True}

        return {"confirmed": False}

    def calculate_stop_loss(self, direction: str) -> float:
        """Calculate initial stop loss from gate levels plus buffer."""
        if direction == "long":
            return (self.gate_low or 0.0) - self.stop_buffer_ticks
        return (self.gate_high or 0.0) + self.stop_buffer_ticks

    def calculate_take_profit(self, entry: float, stop: float, direction: str) -> float:
        """Project take profit at min_risk_reward multiples of risk."""
        risk = abs(entry - stop)
        if risk == 0:
            return entry
        if direction == "long":
            return entry + risk * self.min_risk_reward
        return entry - risk * self.min_risk_reward

    def _apply_trailing(self, row: pd.Series) -> None:
        """RM 1 — Update active_stop via trailing logic every bar."""
        if not self.use_trailing or not self.position_open:
            return
        high = row["High"]
        low = row["Low"]
        if self.position_direction == "long":
            self.trail_extreme = max(self.trail_extreme, high)
            self.active_stop = self.trail_extreme - self.trail_offset_pts
        else:
            self.trail_extreme = min(self.trail_extreme, low)
            self.active_stop = self.trail_extreme + self.trail_offset_pts

    def _apply_break_even(self, row: pd.Series) -> None:
        """RM 2 — Move stop to break-even once profit threshold is reached (once per trade)."""
        if not self.use_break_even or self.be_done or not self.position_open:
            return
        close = row["Close"]
        if self.position_direction == "long":
            if (close - self.entry_price) >= self.be_trigger_pts:
                self.be_done = True
                self.active_stop = self.entry_price + self.be_buffer_pts
        else:
            if (self.entry_price - close) >= self.be_trigger_pts:
                self.be_done = True
                self.active_stop = self.entry_price - self.be_buffer_pts

    def _check_time_exit(self, row: pd.Series) -> Optional[dict]:
        """RM 3 — Close position at or after the configured exit time."""
        if not self.use_time_exit or not self.position_open:
            return None
        bar_time = row.name.time()
        exit_time = time(self.exit_hour, self.exit_minute)
        if bar_time >= exit_time:
            return self._close_position(row["Close"], "time_exit")
        return None

    def _close_position(self, price: float, reason: str) -> dict:
        """Close the open position, reset trade state, re-arm for next trade."""
        if self.position_direction == "long":
            pnl_pts = price - self.entry_price
        else:
            pnl_pts = self.entry_price - price

        result = {
            "signal_type": "exit",
            "direction": self.position_direction,
            "exit_price": price,
            "reason": reason,
            "pnl_pts": pnl_pts,
            "contracts": self.contracts,
        }

        self.position_open = False
        self.position_direction = None
        self.entry_price = None
        self.trail_extreme = None
        self._reset_trade()

        return result

    def _check_position_exit(self, row: pd.Series) -> Optional[dict]:
        """Evaluate RM 3, RM 2, RM 1, then SL/TP in Pine evaluation order."""
        if not self.position_open:
            return None

        # RM 3 — Time exit (highest priority, mirrors Pine's ordering)
        signal = self._check_time_exit(row)
        if signal:
            return signal

        # RM 2 — Break-even (mutates active_stop once)
        self._apply_break_even(row)

        # RM 1 — Trailing (mutates active_stop every bar)
        self._apply_trailing(row)

        # SL / TP check using the (possibly updated) active_stop
        high = row["High"]
        low = row["Low"]
        if self.position_direction == "long":
            if low <= self.active_stop:
                return self._close_position(self.active_stop, "stop_loss")
            if high >= self.active_tp:
                return self._close_position(self.active_tp, "take_profit")
        else:
            if high >= self.active_stop:
                return self._close_position(self.active_stop, "stop_loss")
            if low <= self.active_tp:
                return self._close_position(self.active_tp, "take_profit")

        return None

    def on_data(self, row: pd.Series, df: pd.DataFrame = None) -> Optional[dict]:
        """Process one bar. Returns a signal dict or None.

        row.name must be a pd.Timestamp (DatetimeIndex row).
        df is the full dataset required for confirmation lookups.
        """
        bar_date = row.name.date()

        # New-day reset (mirrors Pine's new_day block; preserves position_open)
        if self._last_date is not None and bar_date != self._last_date:
            self._reset_day()
        self._last_date = bar_date

        # Position is open — check exits before any other logic
        if self.position_open:
            return self._check_position_exit(row)

        # Build gate
        self.detect_gate(row)

        bar_time = row.name.time()
        in_full_session = self._session_start <= bar_time < self._session_end

        if not self.gate_set or not in_full_session:
            return None

        # Breakout detection
        if not self.waiting_for_retest:
            self.detect_breakout(row)

        # Retest + optional confirmation + entry
        if self.waiting_for_retest and self.detect_retest(row):
            if self.use_confirmation and df is not None:
                idx = df.index.get_loc(row.name)
                if not self.check_confirmation(df, idx)["confirmed"]:
                    # Soft touch: keep waiting_for_retest so a later confirmed bar can enter
                    return None

            close = row["Close"]

            if self.retest_direction == "bullish":
                direction = "long"
                stop_px = self.calculate_stop_loss("long")
                tp_px = self.calculate_take_profit(close, stop_px, "long")
                self.position_direction = "long"
                self.entry_price = close
                self.active_stop = stop_px
                self.active_tp = tp_px
                self.be_done = False
                self.trail_extreme = close
                self.position_open = True
                self.waiting_for_retest = False
                return {
                    "signal_type": "entry",
                    "direction": direction,
                    "entry_price": close,
                    "stop_loss": stop_px,
                    "take_profit": tp_px,
                    "contracts": self.contracts,
                }

            if self.retest_direction == "bearish":
                direction = "short"
                stop_px = self.calculate_stop_loss("short")
                tp_px = self.calculate_take_profit(close, stop_px, "short")
                self.position_direction = "short"
                self.entry_price = close
                self.active_stop = stop_px
                self.active_tp = tp_px
                self.be_done = False
                self.trail_extreme = close
                self.position_open = True
                self.waiting_for_retest = False
                return {
                    "signal_type": "entry",
                    "direction": direction,
                    "entry_price": close,
                    "stop_loss": stop_px,
                    "take_profit": tp_px,
                    "contracts": self.contracts,
                }

        return None
