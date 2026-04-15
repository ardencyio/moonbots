"""
Regime-Based Strategy Classes.

Defines trading strategies for each HMM regime with position sizing,
entry/exit logic, and risk parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional



@dataclass
class StrategySignal:
    """Signal returned by strategy."""

    action: str  # "enter_long", "exit", "hold"
    position_size: float  # Fraction of capital (0-1)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


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

    Characteristics:
    - Higher leverage (2x-3x)
    - Smaller stop losses (tight)
    - Trend-following with momentum confirmation
    """

    name = "low_vol_bull"

    def __init__(
        self,
        default_leverage: float = 2.0,
        min_risk_reward: float = 2.0,
        max_position_pct: float = 0.30,
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct

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

    Characteristics:
    - Moderate leverage (1x-1.5x)
    - Wider stop losses
    - More selective entries
    """

    name = "mid_vol_cautious"

    def __init__(
        self,
        default_leverage: float = 1.2,
        min_risk_reward: float = 2.5,
        max_position_pct: float = 0.20,
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct

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

    Characteristics:
    - Low leverage (0.5x-1x) or no position
    - Wide stops, small positions
    - Defensive positioning
    """

    name = "high_vol_defensive"

    def __init__(
        self,
        default_leverage: float = 0.7,
        min_risk_reward: float = 3.0,
        max_position_pct: float = 0.10,
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct

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

    def __init__(self, n_regimes: int):
        self.n_regimes = n_regimes
        self.low_vol_strategy = LowVolBullStrategy()
        self.mid_vol_strategy = MidVolCautiousStrategy()
        self.high_vol_strategy = HighVolDefensiveStrategy()
        self.uncertainty_strategy = HighVolDefensiveStrategy()  # Conservative fallback

    def get_strategy_for_regime(
        self, regime_id: int, regime_infos: dict
    ) -> BaseRegimeStrategy:
        """
        Select strategy based on regime's VOLATILITY RANK.

        Args:
            regime_id: Current regime ID
            regime_infos: Dict of regime_id -> RegimeInfo with volatility

        Returns:
            Appropriate strategy instance
        """
        if regime_id not in regime_infos:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown regime_id: {regime_id}, using uncertainty strategy"
            )
            return self.uncertainty_strategy

        # Get volatility ranking
        regime_volatility = regime_infos[regime_id].expected_volatility

        # Sort all regimes by volatility
        sorted_by_vol = sorted(
            regime_infos.values(),
            key=lambda r: r.expected_volatility,
        )

        # Find where current regime ranks in volatility
        n = len(sorted_by_vol)
        vol_rank = next(
            i for i, r in enumerate(sorted_by_vol) if r.regime_id == regime_id
        )

        volatility_fraction = vol_rank / max(n - 1, 1)

        logger = logging.getLogger(__name__)
        logger.info(
            f"Regime {regime_id} volatility rank: {vol_rank}/{n - 1} "
            f"(vol={regime_volatility * 100:.3f}%, fraction={volatility_fraction:.2f})"
        )

        if volatility_fraction <= 0.33:
            return self.low_vol_strategy
        elif volatility_fraction <= 0.66:
            return self.mid_vol_strategy
        else:
            return self.high_vol_strategy

    def get_uncertainty_strategy(self) -> BaseRegimeStrategy:
        """Return strategy for uncertain/flickering regime."""
        return self.uncertainty_strategy

    def should_enter_position(
        self,
        strategy: BaseRegimeStrategy,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
    ) -> StrategySignal:
        """Generate entry signal using selected strategy."""
        return strategy.generate_signal(
            current_price=current_price,
            volatility=volatility,
            trend=trend,
            momentum=momentum,
            regime_strength=regime_strength,
        )
