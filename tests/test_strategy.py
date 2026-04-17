"""Tests for OpenGateStrategy."""


import pandas as pd
import pytest

from backtest.strategies.open_gate import OpenGateStrategy


@pytest.fixture
def strategy():
    return OpenGateStrategy()


@pytest.fixture
def sample_bars():
    """Create sample 5-minute bars for a single session."""
    dates = pd.date_range("2024-01-15 09:30", periods=10, freq="5min")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 101.5, 100, 100.5, 101, 100.8, 100, 99.5],
            "High": [102, 103, 104, 103, 101, 102, 102.5, 101.5, 100.5, 100],
            "Low":  [99,  100, 101, 100, 99.5, 100, 100.5, 100, 99, 98.5],
            "Close":[101, 102, 103, 100.5, 100, 101.5, 101, 100.5, 99.5, 99],
            "Volume":[1000] * 10,
        },
        index=dates,
    )


class TestGateDetection:
    def test_gate_sets_high_low(self, sample_bars):
        strategy = OpenGateStrategy()
        gate_df = sample_bars.iloc[:2]
        strategy.detect_gate(gate_df)
        assert strategy.gate_set is True
        assert strategy.gate_high == 102.0
        assert strategy.gate_low == 99.0

    def test_empty_gate_fails(self):
        strategy = OpenGateStrategy()
        dates = pd.DatetimeIndex([])
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"], index=dates)
        result = strategy.detect_gate(df)
        assert len(result) == 0
        assert strategy.gate_set is False


class TestBreakoutDetection:
    def test_bullish_breakout(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.detect_gate(sample_bars.iloc[:2])

        bar = pd.Series({"Close": 105.0, "High": 105, "Low": 104})
        strategy.detect_breakout(bar)
        assert strategy.broke_above is True
        assert strategy.waiting_for_retest is True
        assert strategy.retest_direction == "bullish"

    def test_bearish_breakout(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.detect_gate(sample_bars.iloc[:2])

        bar = pd.Series({"Close": 97.0, "High": 100, "Low": 97})
        strategy.detect_breakout(bar)
        assert strategy.broke_below is True
        assert strategy.waiting_for_retest is True
        assert strategy.retest_direction == "bearish"

    def test_no_breakout_without_gate(self):
        strategy = OpenGateStrategy()
        bar = pd.Series({"Close": 105.0, "High": 105, "Low": 104})
        strategy.detect_breakout(bar)
        assert strategy.waiting_for_retest is False


class TestRetestDetection:
    def test_bullish_retest(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.detect_gate(sample_bars.iloc[:2])
        strategy.broke_above = True
        strategy.waiting_for_retest = True
        strategy.retest_direction = "bullish"

        bar = pd.Series({"Close": 102, "High": 103, "Low": 101})
        result = strategy.detect_retest(bar)
        assert result is True

    def test_bearish_retest(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.detect_gate(sample_bars.iloc[:2])
        strategy.broke_below = True
        strategy.waiting_for_retest = True
        strategy.retest_direction = "bearish"

        bar = pd.Series({"Close": 100, "High": 100, "Low": 99})
        result = strategy.detect_retest(bar)
        assert result is True

    def test_no_retest_without_waiting(self):
        strategy = OpenGateStrategy()
        bar = pd.Series({"Close": 100, "High": 100, "Low": 99})
        assert strategy.detect_retest(bar) is False


class TestConfirmation:
    def test_bullish_engulfing(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.retest_direction = "bullish"
        # prev1 bearish (close < open), current bullish that engulfs prev1
        df = pd.DataFrame({
            "Open":  [100, 101, 99],
            "High":  [101, 102, 103],
            "Low":   [99,  99, 98],
            "Close": [100, 98, 102],
            "Volume":[1000, 1000, 1000],
        })
        result = strategy.check_confirmation(df, 2)
        assert result["confirmed"] is True

    def test_bearish_engulfing(self):
        strategy = OpenGateStrategy()
        strategy.retest_direction = "bearish"
        # prev1 bullish (close > open), current bearish that engulfs
        df = pd.DataFrame({
            "Open":  [100, 98, 103],
            "High":  [101, 102, 104],
            "Low":   [99,  98, 96],
            "Close": [101, 101, 97],
            "Volume":[1000, 1000, 1000],
        })
        result = strategy.check_confirmation(df, 2)
        assert result["confirmed"] is True


class TestStopLossTakeProfit:
    def test_long_stop_below_gate_low(self, sample_bars):
        strategy = OpenGateStrategy({"stop_buffer_ticks": 2.0})
        strategy.detect_gate(sample_bars.iloc[:2])
        stop = strategy.calculate_stop_loss("long")
        assert stop < strategy.gate_low
        assert stop == pytest.approx(strategy.gate_low - 2.0)

    def test_short_stop_above_gate_high(self, sample_bars):
        strategy = OpenGateStrategy({"stop_buffer_ticks": 1.5})
        strategy.detect_gate(sample_bars.iloc[:2])
        stop = strategy.calculate_stop_loss("short")
        assert stop > strategy.gate_high
        assert stop == pytest.approx(strategy.gate_high + 1.5)

    def test_take_profit_long(self):
        strategy = OpenGateStrategy({"min_risk_reward": 2.0})
        strategy.position_direction = "long"
        tp = strategy.calculate_take_profit(100, 90)
        assert tp == 120  # risk=10, rr=2 -> 100+20

    def test_take_profit_short(self):
        strategy = OpenGateStrategy({"min_risk_reward": 2.0})
        strategy.position_direction = "short"
        tp = strategy.calculate_take_profit(100, 110)
        assert tp == 80  # risk=10, rr=2 -> 100-20


class TestReset:
    def test_reset_clears_all_state(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.detect_gate(sample_bars.iloc[:2])
        strategy.broke_above = True
        strategy.waiting_for_retest = True
        strategy.reset()

        assert strategy.gate_set is False
        assert strategy.broke_above is False
        assert strategy.waiting_for_retest is False
        assert strategy.retest_direction is None


class TestStatePersistence:
    def test_get_and_set_state(self, sample_bars):
        strategy = OpenGateStrategy()
        strategy.detect_gate(sample_bars.iloc[:2])
        strategy.broke_above = True
        state = strategy.get_state()

        new_strategy = OpenGateStrategy()
        new_strategy.set_state(state)
        assert new_strategy.gate_high == strategy.gate_high
        assert new_strategy.gate_low == strategy.gate_low
        assert new_strategy.gate_set  == strategy.gate_set
        assert new_strategy.broke_above == strategy.broke_above


class TestOnDataConfirmationGate:
    """Verify on_data respects the use_confirmation flag when retest fires."""

    def _build_setup_df(self, confirm_bar: dict) -> pd.DataFrame:
        """Build a 6-bar intraday frame that establishes a gate, breaks above it,
        and then retests it on the final bar — configurable whether that final bar
        is a confirmation candle or a neutral retest."""
        idx = pd.date_range("2024-01-15 09:30", periods=6, freq="1min")
        rows = [
            # bars 0-1: gate = [99, 102]
            {"Open": 100, "High": 102, "Low": 99,  "Close": 101, "Volume": 1000},
            {"Open": 101, "High": 102, "Low": 100, "Close": 101, "Volume": 1000},
            # bar 2: breakout above gate_high (102)
            {"Open": 101, "High": 105, "Low": 101, "Close": 104, "Volume": 1000},
            # bar 3: drift higher
            {"Open": 104, "High": 106, "Low": 103, "Close": 105, "Volume": 1000},
            # bar 4: previous bar for engulfing pattern (bearish red)
            {"Open": 105, "High": 105, "Low": 102, "Close": 103, "Volume": 1000},
            # bar 5: retest — configurable
            confirm_bar,
        ]
        return pd.DataFrame(rows, index=idx)

    def test_retest_without_confirmation_blocks_entry(self):
        # Neutral retest: low touches gate_high but there's no reaction body.
        neutral = {"Open": 102, "High": 102, "Low": 101.5, "Close": 102, "Volume": 1000}
        df = self._build_setup_df(neutral)
        strategy = OpenGateStrategy({"gate_candle_minutes": 2, "use_confirmation": True})
        strategy.detect_gate(df.iloc[:2])

        # Step through bars 2..5; retest fires on bar 5 but confirmation must block the entry
        signals = []
        for i in range(2, len(df)):
            sig = strategy.on_data(df.iloc[i], df)
            if sig is not None:
                signals.append(sig)

        assert signals == [], f"expected no entry when confirmation missing; got {signals}"
        assert strategy.waiting_for_retest is True
        assert strategy.position_open is False

    def test_retest_with_bullish_engulfing_triggers_entry(self):
        # Bar 4 was red; this bar opens below prev close and closes above prev open — bullish engulfing.
        engulfing = {"Open": 102, "High": 106, "Low": 102, "Close": 106, "Volume": 1000}
        df = self._build_setup_df(engulfing)
        strategy = OpenGateStrategy({"gate_candle_minutes": 2, "use_confirmation": True})
        strategy.detect_gate(df.iloc[:2])

        signals = []
        for i in range(2, len(df)):
            sig = strategy.on_data(df.iloc[i], df)
            if sig is not None:
                signals.append(sig)

        entries = [s for s in signals if s["signal_type"] == "entry"]
        assert len(entries) == 1, f"expected one entry; got {signals}"
        assert entries[0]["direction"] == "long"
        assert strategy.position_open is True

    def test_multiple_neutral_retests_then_confirmation_triggers_entry(self):
        """ORB invariant: a soft-touch retest must not cancel the setup — the
        strategy keeps waiting_for_retest=True across multiple unreactive
        bars and enters only when a confirmation candle finally prints."""
        idx = pd.date_range("2024-01-15 09:30", periods=8, freq="1min")
        rows = [
            # bars 0-1: gate = [99, 102]
            {"Open": 100, "High": 102, "Low": 99,  "Close": 101, "Volume": 1000},
            {"Open": 101, "High": 102, "Low": 100, "Close": 101, "Volume": 1000},
            # bar 2: breakout above gate_high (102)
            {"Open": 101, "High": 105, "Low": 101, "Close": 104, "Volume": 1000},
            # bar 3: neutral retest — low touches gate_high, but no reaction body
            {"Open": 103, "High": 104, "Low": 102, "Close": 103, "Volume": 1000},
            # bar 4: another neutral retest — still no reaction
            {"Open": 103, "High": 103.5, "Low": 102, "Close": 103, "Volume": 1000},
            # bar 5: bearish candle that sets up the engulfing on bar 6
            {"Open": 104, "High": 104, "Low": 101.5, "Close": 102, "Volume": 1000},
            # bar 6: prior bar for engulfing (also bearish)
            {"Open": 103, "High": 103.5, "Low": 101.5, "Close": 102, "Volume": 1000},
            # bar 7: bullish engulfing — opens < prev close, closes > prev open,
            # low touches gate_high → this is the entry bar
            {"Open": 101.5, "High": 106, "Low": 101.5, "Close": 106, "Volume": 1000},
        ]
        df = pd.DataFrame(rows, index=idx)
        strategy = OpenGateStrategy({"gate_candle_minutes": 2, "use_confirmation": True})
        strategy.detect_gate(df.iloc[:2])

        signals = []
        for i in range(2, len(df)):
            sig = strategy.on_data(df.iloc[i], df)
            if sig is not None:
                signals.append(sig)

        # Invariant: across the two neutral retest bars (3 and 4), waiting_for_retest
        # must never flip to False — the breakout setup must survive soft touches.
        entries = [s for s in signals if s["signal_type"] == "entry"]
        assert len(entries) == 1, f"expected exactly one entry; got {signals}"
        assert entries[0]["direction"] == "long"
        assert strategy.position_open is True
