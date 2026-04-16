"""
Tests for the regime-based strategy layer.

Covers:
- Vol-rank → strategy mapping for 3/4/5 regime cases
- Uncertainty mode: halves position size AND forces leverage = 1.0x
- Rebalance threshold: 10% trigger and zero-to-zero no-op guard
- LABEL_TO_STRATEGY resolution for every spec label
- update_regime_infos re-mapping after retrain
"""

from __future__ import annotations

from datetime import datetime

import pytest

from shared.core.hmm.hmm_engine import RegimeInfo
from shared.core.hmm.regime_strategies import (
    LABEL_TO_STRATEGY,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MidVolCautiousStrategy,
    Signal,
    StrategyOrchestrator,
)


def _info(regime_id: int, name: str, vol: float) -> RegimeInfo:
    return RegimeInfo(
        regime_id=regime_id,
        regime_name=name,
        expected_return=0.0,
        expected_volatility=vol,
    )


class TestLabelToStrategy:
    """LABEL_TO_STRATEGY dict must cover every label the HMM can emit."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("CRASH", HighVolDefensiveStrategy),
            ("STRONG_BEAR", HighVolDefensiveStrategy),
            ("BEAR", HighVolDefensiveStrategy),
            ("WEAK_BEAR", MidVolCautiousStrategy),
            ("NEUTRAL", MidVolCautiousStrategy),
            ("WEAK_BULL", MidVolCautiousStrategy),
            ("BULL", LowVolBullStrategy),
            ("STRONG_BULL", LowVolBullStrategy),
            ("EUPHORIA", LowVolBullStrategy),
        ],
    )
    def test_each_label_maps_to_correct_strategy(self, label, expected):
        assert LABEL_TO_STRATEGY[label] is expected


class TestVolRankMapping:
    """StrategyOrchestrator must pick strategies by volatility rank."""

    def test_three_regimes_map_by_vol(self):
        orch = StrategyOrchestrator(n_regimes=3)
        regime_infos = {
            0: _info(0, "BEAR", vol=0.03),
            1: _info(1, "NEUTRAL", vol=0.015),
            2: _info(2, "BULL", vol=0.008),
        }
        orch.update_regime_infos(regime_infos)
        # Sorted asc by vol → BULL(low), NEUTRAL(mid), BEAR(high)
        assert isinstance(
            orch.get_strategy_for_regime(2, regime_infos), LowVolBullStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(1, regime_infos), MidVolCautiousStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(0, regime_infos), HighVolDefensiveStrategy
        )

    def test_four_regimes_partition(self):
        orch = StrategyOrchestrator(n_regimes=4)
        regime_infos = {
            0: _info(0, "CRASH", vol=0.05),
            1: _info(1, "BEAR", vol=0.025),
            2: _info(2, "BULL", vol=0.012),
            3: _info(3, "EUPHORIA", vol=0.008),
        }
        orch.update_regime_infos(regime_infos)
        # Ascending vol: 3 (0.0), 2 (0.333), 1 (0.666), 0 (1.0)
        # Bucket thresholds: <= 0.33 → low; <= 0.66 → mid; else → high
        assert isinstance(
            orch.get_strategy_for_regime(3, regime_infos), LowVolBullStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(2, regime_infos), MidVolCautiousStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(1, regime_infos), HighVolDefensiveStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(0, regime_infos), HighVolDefensiveStrategy
        )

    def test_five_regimes_partition(self):
        orch = StrategyOrchestrator(n_regimes=5)
        regime_infos = {
            0: _info(0, "CRASH", vol=0.06),
            1: _info(1, "BEAR", vol=0.04),
            2: _info(2, "NEUTRAL", vol=0.02),
            3: _info(3, "BULL", vol=0.012),
            4: _info(4, "EUPHORIA", vol=0.006),
        }
        orch.update_regime_infos(regime_infos)
        # Ascending vol: 4 (0.0), 3 (0.25), 2 (0.5), 1 (0.75), 0 (1.0)
        assert isinstance(
            orch.get_strategy_for_regime(4, regime_infos), LowVolBullStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(3, regime_infos), LowVolBullStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(2, regime_infos), MidVolCautiousStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(1, regime_infos), HighVolDefensiveStrategy
        )
        assert isinstance(
            orch.get_strategy_for_regime(0, regime_infos), HighVolDefensiveStrategy
        )

    def test_cache_populated_on_first_call(self):
        """get_strategy_for_regime must populate the cache on first use."""
        orch = StrategyOrchestrator(n_regimes=3)
        regime_infos = {
            0: _info(0, "BEAR", vol=0.03),
            1: _info(1, "NEUTRAL", vol=0.015),
            2: _info(2, "BULL", vol=0.008),
        }
        assert not orch._cached_strategy_map
        # First call should populate without a prior update_regime_infos
        strategy = orch.get_strategy_for_regime(2, regime_infos)
        assert isinstance(strategy, LowVolBullStrategy)
        assert orch._cached_strategy_map, "cache must be populated on first call"


class TestUncertaintyMode:
    """Uncertainty mode halves size AND clamps leverage to 1.0x."""

    def _enter_signal_inputs(self, strategy):
        return {
            "strategy": strategy,
            "symbol": "SPY",
            "current_price": 100.0,
            "volatility": 0.01,
            "trend": 0.10,  # strong trend so low-vol bull will enter_long
            "momentum": 0.05,
            "regime_strength": 0.9,
            "regime_id": 0,
            "regime_name": "BULL",
            "regime_probability": 0.95,
            "atr": 1.0,
            "ema50": 95.0,
            "timestamp": datetime(2026, 1, 1),
        }

    def test_uncertainty_halves_size_and_clamps_leverage(self):
        orch = StrategyOrchestrator(n_regimes=3)
        strategy = LowVolBullStrategy()

        # Full confidence: leverage stays at 1.25, size not halved
        full = orch.should_enter_position(
            **self._enter_signal_inputs(strategy),
            confidence=0.9,
            is_flickering=False,
            min_confidence=0.55,
        )
        assert isinstance(full, Signal)
        assert full.leverage == pytest.approx(1.25)

        # Low confidence: leverage clamped to 1.0, size halved vs full
        uncertain = orch.should_enter_position(
            **self._enter_signal_inputs(strategy),
            confidence=0.30,
            is_flickering=False,
            min_confidence=0.55,
        )
        assert uncertain.leverage == pytest.approx(1.0)
        assert uncertain.position_size_pct == pytest.approx(
            full.position_size_pct * 0.5
        )
        assert "UNCERTAINTY" in uncertain.reasoning

    def test_flickering_triggers_uncertainty(self):
        orch = StrategyOrchestrator(n_regimes=3)
        flicker = orch.should_enter_position(
            **self._enter_signal_inputs(LowVolBullStrategy()),
            confidence=0.95,
            is_flickering=True,
            min_confidence=0.55,
        )
        assert flicker.leverage == pytest.approx(1.0)
        assert "UNCERTAINTY" in flicker.reasoning


class TestSignalShape:
    """Signal carries all spec Phase-3 fields."""

    def test_signal_fields_populated(self):
        strategy = LowVolBullStrategy()
        sig = strategy.generate_signal(
            symbol="SPY",
            current_price=100.0,
            volatility=0.01,
            trend=0.1,
            momentum=0.05,
            regime_strength=0.9,
            regime_id=2,
            regime_name="BULL",
            regime_probability=0.92,
            atr=1.0,
            ema50=95.0,
            timestamp=datetime(2026, 1, 1),
        )
        assert isinstance(sig, Signal)
        assert sig.symbol == "SPY"
        assert sig.direction == "long"
        assert sig.strategy_name == "low_vol_bull"
        assert sig.regime_id == 2
        assert sig.regime_name == "BULL"
        assert sig.regime_probability == pytest.approx(0.92)
        assert sig.leverage == pytest.approx(1.25)
        assert sig.position_size_pct <= 0.95
        assert sig.timestamp == datetime(2026, 1, 1)

    def test_mid_vol_trend_intact_allocation(self):
        strategy = MidVolCautiousStrategy()
        sig = strategy.generate_signal(
            symbol="SPY",
            current_price=100.0,
            volatility=0.02,
            trend=0.1,
            momentum=0.05,
            regime_strength=0.8,
            regime_id=1,
            regime_name="NEUTRAL",
            regime_probability=0.8,
            atr=1.0,
            ema50=95.0,  # price > ema50 → trend intact
            timestamp=datetime(2026, 1, 1),
        )
        assert sig.leverage == pytest.approx(1.0)
        assert sig.position_size_pct <= 0.95

    def test_mid_vol_trend_broken_allocation(self):
        strategy = MidVolCautiousStrategy()
        sig = strategy.generate_signal(
            symbol="SPY",
            current_price=100.0,
            volatility=0.02,
            trend=0.1,
            momentum=0.05,
            regime_strength=0.8,
            regime_id=1,
            regime_name="NEUTRAL",
            regime_probability=0.8,
            atr=1.0,
            ema50=105.0,  # price < ema50 → trend broken
            timestamp=datetime(2026, 1, 1),
        )
        assert sig.leverage == pytest.approx(1.0)
        assert sig.position_size_pct <= 0.60

    def test_high_vol_defensive_caps(self):
        strategy = HighVolDefensiveStrategy()
        sig = strategy.generate_signal(
            symbol="SPY",
            current_price=100.0,
            volatility=0.04,
            trend=0.2,
            momentum=0.1,
            regime_strength=0.9,
            regime_id=0,
            regime_name="CRASH",
            regime_probability=0.8,
            atr=2.0,
            ema50=90.0,
            timestamp=datetime(2026, 1, 1),
        )
        assert sig.leverage == pytest.approx(1.0)
        assert sig.position_size_pct <= 0.60


class TestRebalance:
    """10% rebalancing threshold with zero-to-zero no-op guard."""

    def test_zero_to_zero_no_rebalance(self):
        orch = StrategyOrchestrator(n_regimes=3)
        orch.update_position_size(0.0)
        assert orch.needs_rebalance(0.0) is False

    def test_first_open_rebalances(self):
        orch = StrategyOrchestrator(n_regimes=3)
        orch.update_position_size(0.0)
        assert orch.needs_rebalance(0.50) is True

    def test_small_change_skips(self):
        orch = StrategyOrchestrator(n_regimes=3)
        orch.update_position_size(0.50)
        assert orch.needs_rebalance(0.52) is False  # 4% change, below 10% threshold

    def test_large_change_triggers(self):
        orch = StrategyOrchestrator(n_regimes=3)
        orch.update_position_size(0.50)
        assert orch.needs_rebalance(0.60) is True  # 20% change


class TestUpdateRegimeInfosAfterRetrain:
    """After a retrain, the strategy map must be rebuilt for the new regimes."""

    def test_remap_after_retrain_changes_cache(self):
        orch = StrategyOrchestrator(n_regimes=3)
        first = {
            0: _info(0, "BEAR", vol=0.03),
            1: _info(1, "NEUTRAL", vol=0.015),
            2: _info(2, "BULL", vol=0.008),
        }
        orch.update_regime_infos(first)
        low_first = orch.get_strategy_for_regime(2, first)
        assert isinstance(low_first, LowVolBullStrategy)

        # Retrain produces different regime_ids mapped to different vol levels
        second = {
            0: _info(0, "BULL", vol=0.009),  # now id 0 is lowest vol
            1: _info(1, "NEUTRAL", vol=0.020),
            2: _info(2, "BEAR", vol=0.035),  # id 2 is now highest vol
        }
        orch.update_regime_infos(second)
        assert isinstance(orch.get_strategy_for_regime(0, second), LowVolBullStrategy)
        assert isinstance(
            orch.get_strategy_for_regime(2, second), HighVolDefensiveStrategy
        )

    def test_get_strategy_updates_when_regime_infos_shift(self):
        orch = StrategyOrchestrator(n_regimes=3)
        first = {
            0: _info(0, "BEAR", vol=0.03),
            1: _info(1, "NEUTRAL", vol=0.015),
            2: _info(2, "BULL", vol=0.008),
        }
        orch.get_strategy_for_regime(2, first)

        # Provide a shifted copy of regime_infos (different vols)
        shifted = {
            0: _info(0, "BULL", vol=0.008),
            1: _info(1, "NEUTRAL", vol=0.015),
            2: _info(2, "BEAR", vol=0.03),
        }
        # Regime 0 (lowest vol now) should resolve as low-vol strategy
        strat = orch.get_strategy_for_regime(0, shifted)
        assert isinstance(strat, LowVolBullStrategy)
