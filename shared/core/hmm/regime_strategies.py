"""
Regime-Based Strategy Classes.

Defines trading strategies for each HMM regime with position sizing,
entry/exit logic, and risk parameters.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# LABEL_TO_STRATEGY mapping for regime label -> strategy selection
# Used by the strategy orchestrator to select appropriate strategy based on regime label
LABEL_TO_STRATEGY = {
    "BULL_BULL": "low_vol",
    "BULL_NEUTRAL": "low_vol",
    "BULL_BEAR": "low_vol",
    "NEUTRAL_BULL": "mid_vol",
    "NEUTRAL_NEUTRAL": "mid_vol",
    "NEUTRAL_BEAR": "mid_vol",
    "BEAR_BULL": "high_vol",
    "BEAR_NEUTRAL": "high_vol",
    "BEAR_BEAR": "high_vol",
}


@dataclass
class StrategySignal:
    """Signal returned by strategy."""

    action: str  # "enter_long", "exit", "hold"
    position_size: float  # Fraction of capital (0-1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


@dataclass
class Signal:
    """
    Trading signal with full risk parameters.

    Matches spec Phase 3 Signal dataclass.
    """

    symbol: str
    direction: str  # "long", "short", "flat"
    confidence: float  # Regime probability (0-1)
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float  # Position as % of capital
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: datetime
    reasoning: str
    strategy_name: str
    metadata: Optional[dict] = None


class BaseRegimeStrategy(ABC):
    """Base class for regime-specific strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signal(
        self,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
    ) -> StrategySignal:
        """Generate trading signal based on market conditions."""
        ...

    @abstractmethod
    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        """Calculate position size given risk parameters."""
        ...


class LowVolBullStrategy(BaseRegimeStrategy):
    """
    Strategy for low volatility bullish regimes.

    Spec: max_asset_allocation=0.95, leverage=1.25

    Characteristics:
    - Max 95% asset allocation
    - 1.25x leverage
    - Tight stops (1.5x volatility)
    - Trend-following with momentum confirmation
    """

    name = "low_vol_bull"

    def __init__(
        self,
        default_leverage: float = 1.25,
        min_risk_reward: float = 2.0,
        max_position_pct: float = 0.95,
        stop_mult: float = 0.5,  # ATR multiplier for stop (0.5 for low vol)
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.stop_mult = stop_mult

    def generate_signal(
        self,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
    ) -> StrategySignal:
        # Strong trend + momentum = enter long
        if trend > 0.05 and momentum > 0.02:
            action = "enter_long"
            reason = "Bull trend with momentum confirmation"
        # Weak signals = hold
        elif abs(trend) < 0.02 and abs(momentum) < 0.01:
            action = "hold"
            reason = "Low conviction signals"
        else:
            action = "hold"
            reason = "Waiting for setup"

        position_size = self.calculate_position_size(
            capital=1.0,
            stop_loss_distance_pct=volatility * 1.5,
            max_position_size_pct=self.max_position_pct,
        )

        stop_loss = current_price * (1 - volatility * 1.5)
        take_profit = current_price * (1 + volatility * 1.5 * self.min_risk_reward)

        return StrategySignal(
            action=action,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        # Risk-based sizing: position = capital * leverage / (stop_distance * risk_factor)
        risk_per_trade = 0.02  # Risk 2% per trade
        position_size = min(
            (risk_per_trade / max(stop_loss_distance_pct, 0.001))
            * self.default_leverage,
            max_position_size_pct,
        )
        return position_size


class MidVolCautiousStrategy(BaseRegimeStrategy):
    """
    Strategy for moderate volatility regimes.

    Spec: max_asset_allocation=0.95 (equity) / 0.60 (cash), leverage=1.0

    Characteristics:
    - 95% equity / 60% cash max allocation
    - 1x leverage
    - Wider stops (2.0x volatility)
    - More selective entries
    """

    name = "mid_vol_cautious"

    def __init__(
        self,
        default_leverage: float = 1.0,
        min_risk_reward: float = 2.5,
        max_position_pct: float = 0.95,
        conservative_position_pct: float = 0.60,
        stop_mult: float = 1.0,  # ATR multiplier for stop (1.0 for mid/high vol)
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.conservative_position_pct = conservative_position_pct
        self.stop_mult = stop_mult

    def generate_signal(
        self,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
    ) -> StrategySignal:
        # Only enter on strong setups
        if trend > 0.08 and momentum > 0.03:
            action = "enter_long"
            reason = "Strong trend and momentum"
        else:
            action = "hold"
            reason = "Too weak for mid-vol regime"

        position_size = self.calculate_position_size(
            capital=1.0,
            stop_loss_distance_pct=volatility * 2.0,
            max_position_size_pct=self.max_position_pct,
        )

        stop_loss = current_price * (1 - volatility * 2.0)
        take_profit = current_price * (1 + volatility * 2.0 * self.min_risk_reward)

        return StrategySignal(
            action=action,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        risk_per_trade = 0.015  # Risk 1.5% per trade
        position_size = min(
            (risk_per_trade / max(stop_loss_distance_pct, 0.001))
            * self.default_leverage,
            max_position_size_pct,
        )
        return position_size


class HighVolDefensiveStrategy(BaseRegimeStrategy):
    """
    Strategy for high volatility regimes.

    Spec: max_asset_allocation=0.60, leverage=1.0

    Characteristics:
    - Max 60% asset allocation (high cash reserve)
    - 1x leverage
    - Wide stops (2.5x volatility)
    - Defensive positioning
    """

    name = "high_vol_defensive"

    def __init__(
        self,
        default_leverage: float = 1.0,
        min_risk_reward: float = 3.0,
        max_position_pct: float = 0.60,
        stop_mult: float = 1.0,  # ATR multiplier for stop
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.stop_mult = stop_mult

    def generate_signal(
        self,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
    ) -> StrategySignal:
        # Very selective in high vol
        if trend > 0.12 and momentum > 0.05 and regime_strength > 0.75:
            action = "enter_long"
            reason = "Exceptional setup in high vol"
        elif trend < -0.05 or momentum < -0.03:
            action = "exit"
            reason = "Deteriorating conditions"
        else:
            action = "hold"
            reason = "Defensive stance in high vol"

        position_size = self.calculate_position_size(
            capital=1.0,
            stop_loss_distance_pct=volatility * 2.5,
            max_position_size_pct=self.max_position_pct,
        )

        stop_loss = current_price * (1 - volatility * 2.5)
        take_profit = current_price * (1 + volatility * 2.5 * self.min_risk_reward)

        return StrategySignal(
            action=action,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        risk_per_trade = 0.01  # Risk 1% per trade
        position_size = min(
            (risk_per_trade / max(stop_loss_distance_pct, 0.001))
            * self.default_leverage,
            max_position_size_pct,
        )
        return position_size


class StrategyOrchestrator:
    """
    Maps HMM regimes to strategies based on VOLATILITY RANKING.

    Key principle: The STRATEGY is chosen by VOLATILITY, not by regime label.
    Regime labels (BEAR/NEUTRAL/BULL) are for display only.
    Strategy selection sorts regimes by their expected volatility.
    """

    # Backward-compat: label aliases for different regime counts
    # Maps label strings to strategy types
    LABEL_TO_STRATEGY: dict[str, str] = {
        # 3-regime
        "BEAR": "high_vol_defensive",
        "NEUTRAL": "mid_vol_cautious",
        "BULL": "low_vol_bull",
        # 4-regime
        "CRASH": "high_vol_defensive",
        "EUPHORIA": "low_vol_bull",
        # 5-regime
        "STRONG_BEAR": "high_vol_defensive",
        "WEAK_BEAR": "mid_vol_cautious",
        "WEAK_BULL": "mid_vol_cautious",
        "STRONG_BULL": "low_vol_bull",
    }

    def __init__(self, n_regimes: int):
        self.n_regimes = n_regimes
        self.low_vol_strategy = LowVolBullStrategy()
        self.mid_vol_strategy = MidVolCautiousStrategy()
        self.high_vol_strategy = HighVolDefensiveStrategy()
        self.uncertainty_strategy = HighVolDefensiveStrategy()  # Conservative fallback
        self._cached_strategy_map: dict[int, BaseRegimeStrategy] = {}
        self._last_regime_infos: dict = {}
        self.rebalance_threshold = 0.10  # 10% threshold for rebalancing
        self.last_position_size = 0.0  # Track last position size

    def update_regime_infos(self, regime_infos: dict):
        """
        Update regime info mapping and cache strategy assignments.

        Args:
            regime_infos: Dict mapping regime_id -> RegimeInfo

        Returns:
            Updated cached strategy map
        """
        if not regime_infos:
            return self._cached_strategy_map

        self._last_regime_infos = regime_infos
        sorted_by_vol = sorted(
            regime_infos.values(),
            key=lambda r: r.expected_volatility,
        )
        n = len(sorted_by_vol)

        # Build cached strategy map
        # Use int() to avoid np.int64 vs int key mismatch
        self._cached_strategy_map = {}
        for i, regime_info in enumerate(sorted_by_vol):
            volatility_fraction = i / max(n - 1, 1)
            regime_id = int(regime_info.regime_id)
            if volatility_fraction <= 0.33:
                self._cached_strategy_map[regime_id] = self.low_vol_strategy
            elif volatility_fraction <= 0.66:
                self._cached_strategy_map[regime_id] = self.mid_vol_strategy
            else:
                self._cached_strategy_map[regime_id] = self.high_vol_strategy

        logger = logging.getLogger(__name__)
        logger.info(
            f"Updated strategy map for {n} regimes: {self._cached_strategy_map}"
        )
        return self._cached_strategy_map

    def get_strategy_for_regime(
        self, regime_id: int, regime_infos: dict
    ) -> BaseRegimeStrategy:
        """
        Select strategy based on regime's VOLATILITY RANK.

        Uses cached mapping if regime_infos match last update.

        Args:
            regime_id: Current regime ID
            regime_infos: Dict of regime_id -> RegimeInfo with volatility

        Returns:
            Appropriate strategy instance
        """
        # Populate cache if empty or regime_infos changed
        cache_empty = not self._cached_strategy_map
        cache_stale = regime_infos != self._last_regime_infos
        if cache_empty or cache_stale:
            self.update_regime_infos(regime_infos)

        # Return cached strategy or default
        if regime_id in self._cached_strategy_map:
            return self._cached_strategy_map[regime_id]

        logger = logging.getLogger(__name__)
        logger.warning(f"Unknown regime_id: {regime_id}, using uncertainty strategy")
        return self.uncertainty_strategy

    def should_enter_position(
        self,
        strategy: BaseRegimeStrategy,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
        confidence: float = 1.0,
        is_flickering: bool = False,
        min_confidence: float = 0.55,
    ) -> StrategySignal:
        """
        Generate entry signal using selected strategy.

        Includes uncertainty mode logic:
        - Reduce position sizes by 50% when confidence < min_confidence or flickering

        Args:
            strategy: Selected regime strategy
            current_price: Current asset price
            volatility: Current volatility
            trend: Trend strength
            momentum: Momentum indicator
            regime_strength: Regime probability
            confidence: Regime confidence (0-1)
            is_flickering: True if regime is flickering
            min_confidence: Minimum confidence threshold for full sizing

        Returns:
            StrategySignal with potentially reduced size in uncertainty mode
        """
        signal = strategy.generate_signal(
            current_price=current_price,
            volatility=volatility,
            trend=trend,
            momentum=momentum,
            regime_strength=regime_strength,
        )

        # Uncertainty mode: halve position, force 1.0x leverage
        is_uncertain = confidence < min_confidence or is_flickering
        if is_uncertain:
            signal.position_size *= 0.5
            signal.reason = f"{signal.reason} [UNCERTAINTY — size halved]"

        return signal

    def needs_rebalance(self, target_position_size: float) -> bool:
        """
        Check if position size has changed enough to warrant rebalancing.

        Uses 10% threshold to avoid excessive rebalancing.

        Args:
            target_position_size: Proposed new position size

        Returns:
            True if rebalancing is needed
        """
        if self.last_position_size == 0.0:
            return True  # First trade always rebalances

        pct_change = abs(target_position_size - self.last_position_size) / max(
            self.last_position_size, 0.001
        )
        return pct_change > self.rebalance_threshold

    def update_position_size(self, position_size: float) -> None:
        """Update last position size after rebalancing."""
        self.last_position_size = position_size

    def get_uncertainty_strategy(self) -> BaseRegimeStrategy:
        """Return strategy for uncertain/flickering regime."""
        return self.uncertainty_strategy
