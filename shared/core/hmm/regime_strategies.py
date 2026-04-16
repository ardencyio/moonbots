"""
Regime-Based Strategy Classes.

Defines trading strategies for each HMM regime with position sizing,
entry/exit logic, and risk parameters.

DESIGN PRINCIPLE: ALWAYS LONG. NEVER SHORT.
The HMM detects volatility environments, not price direction.
The correct response to high volatility is REDUCING allocation, not reversing direction.
Strategy selection is by VOLATILITY RANK only — regime labels are for display only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    Trading signal with full risk parameters (spec Phase 3 shape).

    Direction is ALWAYS "long" or "flat" — NEVER "short".
    """

    symbol: str
    direction: Literal["long", "flat"]  # ALWAYS LONG. NEVER SHORT.
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
    metadata: dict = field(default_factory=dict)


class BaseRegimeStrategy(ABC):
    """Base class for regime-specific strategies."""

    name: str = "base"
    default_leverage: float = 1.0
    max_position_pct: float = 1.0

    @abstractmethod
    def generate_signal(
        self,
        *,
        symbol: str,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
        regime_id: int,
        regime_name: str,
        regime_probability: float,
        atr: float = 0.0,
        ema50: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Signal:
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

    def _build_signal(
        self,
        *,
        symbol: str,
        action: str,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size_pct: float,
        leverage: float,
        regime_id: int,
        regime_name: str,
        regime_probability: float,
        timestamp: Optional[datetime],
        reasoning: str,
        metadata: Optional[dict] = None,
    ) -> Signal:
        direction: Literal["long", "flat"] = (
            "long" if action == "enter_long" else "flat"
        )
        if action != "enter_long":
            position_size_pct = 0.0
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            leverage=leverage,
            regime_id=regime_id,
            regime_name=regime_name,
            regime_probability=regime_probability,
            timestamp=timestamp if timestamp is not None else datetime.now(),
            reasoning=reasoning,
            strategy_name=self.name,
            metadata=metadata or {"action": action},
        )


class LowVolBullStrategy(BaseRegimeStrategy):
    """
    Low-volatility bullish regimes.

    Spec: max_asset_allocation=0.95, leverage=1.25.
    Stop: max(price - 3*ATR, ema50 - 0.5*ATR).
    """

    name = "low_vol_bull"

    def __init__(
        self,
        default_leverage: float = 1.25,
        min_risk_reward: float = 2.0,
        max_position_pct: float = 0.95,
        stop_mult: float = 3.0,
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.stop_mult = stop_mult

    def generate_signal(
        self,
        *,
        symbol: str,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
        regime_id: int,
        regime_name: str,
        regime_probability: float,
        atr: float = 0.0,
        ema50: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Signal:
        if trend > 0.05 and momentum > 0.02:
            action = "enter_long"
            reason = "Bull trend with momentum confirmation"
        elif abs(trend) < 0.02 and abs(momentum) < 0.01:
            action = "hold"
            reason = "Low conviction signals"
        else:
            action = "hold"
            reason = "Waiting for setup"

        atr = atr if atr > 0 else current_price * volatility
        stop_price_atr = current_price - self.stop_mult * atr
        stop_price_ema = ema50 - 0.5 * atr if ema50 > 0 else stop_price_atr
        stop_loss = max(stop_price_atr, stop_price_ema)
        stop_distance_pct = (current_price - stop_loss) / current_price

        position_size_pct = self.calculate_position_size(
            capital=1.0,
            stop_loss_distance_pct=stop_distance_pct,
            max_position_size_pct=self.max_position_pct,
        )
        take_profit = current_price + (current_price - stop_loss) * self.min_risk_reward

        return self._build_signal(
            symbol=symbol,
            action=action,
            confidence=regime_probability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            leverage=self.default_leverage,
            regime_id=regime_id,
            regime_name=regime_name,
            regime_probability=regime_probability,
            timestamp=timestamp,
            reasoning=reason,
        )

    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        risk_per_trade = 0.02
        position_size = min(
            (risk_per_trade / max(stop_loss_distance_pct, 0.001))
            * self.default_leverage,
            max_position_size_pct,
        )
        return position_size


class MidVolCautiousStrategy(BaseRegimeStrategy):
    """
    Moderate-volatility regimes.

    Spec:
        - price > 50 EMA → allocation 0.95, leverage 1.0x (trend intact).
        - price < 50 EMA → allocation 0.60, leverage 1.0x (trend broken).
        - Stop: ema50 - 0.5*ATR (intact) / ema50 - 1.0*ATR (broken).
    """

    name = "mid_vol_cautious"

    def __init__(
        self,
        default_leverage: float = 1.0,
        min_risk_reward: float = 2.5,
        max_position_pct: float = 0.95,
        conservative_position_pct: float = 0.60,
        stop_mult: float = 1.0,
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.conservative_position_pct = conservative_position_pct
        self.stop_mult = stop_mult

    def generate_signal(
        self,
        *,
        symbol: str,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
        regime_id: int,
        regime_name: str,
        regime_probability: float,
        atr: float = 0.0,
        ema50: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Signal:
        atr = atr if atr > 0 else current_price * volatility

        if ema50 > 0 and current_price > ema50:
            allocation_cap = self.max_position_pct
            stop_loss = ema50 - 0.5 * atr
            trend_label = "trend intact"
        elif ema50 > 0:
            allocation_cap = self.conservative_position_pct
            stop_loss = ema50 - 1.0 * atr
            trend_label = "trend broken"
        else:
            allocation_cap = self.conservative_position_pct
            stop_loss = current_price - self.stop_mult * atr
            trend_label = "no EMA"

        stop_distance_pct = (current_price - stop_loss) / current_price

        if trend > 0.08 and momentum > 0.03:
            action = "enter_long"
            reason = f"Strong trend and momentum ({trend_label})"
        else:
            action = "hold"
            reason = f"Too weak for mid-vol regime ({trend_label})"

        position_size_pct = min(
            self.calculate_position_size(
                capital=1.0,
                stop_loss_distance_pct=stop_distance_pct,
                max_position_size_pct=allocation_cap,
            ),
            allocation_cap,
        )
        take_profit = current_price + (current_price - stop_loss) * self.min_risk_reward

        return self._build_signal(
            symbol=symbol,
            action=action,
            confidence=regime_probability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            leverage=self.default_leverage,
            regime_id=regime_id,
            regime_name=regime_name,
            regime_probability=regime_probability,
            timestamp=timestamp,
            reasoning=reason,
            metadata={"action": action, "trend_label": trend_label},
        )

    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        risk_per_trade = 0.015
        position_size = min(
            (risk_per_trade / max(stop_loss_distance_pct, 0.001))
            * self.default_leverage,
            max_position_size_pct,
        )
        return position_size


class HighVolDefensiveStrategy(BaseRegimeStrategy):
    """
    High-volatility regimes.

    Spec: max_asset_allocation=0.60, leverage=1.0, ALWAYS LONG.
    Stop: ema50 - 1.0*ATR.
    """

    name = "high_vol_defensive"

    def __init__(
        self,
        default_leverage: float = 1.0,
        min_risk_reward: float = 3.0,
        max_position_pct: float = 0.60,
        stop_mult: float = 1.0,
    ):
        self.default_leverage = default_leverage
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = max_position_pct
        self.stop_mult = stop_mult

    def generate_signal(
        self,
        *,
        symbol: str,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
        regime_id: int,
        regime_name: str,
        regime_probability: float,
        atr: float = 0.0,
        ema50: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Signal:
        atr = atr if atr > 0 else current_price * volatility

        if ema50 > 0:
            stop_loss = ema50 - self.stop_mult * atr
        else:
            stop_loss = current_price - self.stop_mult * atr

        stop_distance_pct = (current_price - stop_loss) / current_price

        if trend > 0.12 and momentum > 0.05 and regime_strength > 0.75:
            action = "enter_long"
            reason = "Exceptional setup in high vol"
        else:
            action = "hold"
            reason = "Defensive stance in high vol"

        position_size_pct = self.calculate_position_size(
            capital=1.0,
            stop_loss_distance_pct=stop_distance_pct,
            max_position_size_pct=self.max_position_pct,
        )
        take_profit = current_price + (current_price - stop_loss) * self.min_risk_reward

        return self._build_signal(
            symbol=symbol,
            action=action,
            confidence=regime_probability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size_pct,
            leverage=self.default_leverage,
            regime_id=regime_id,
            regime_name=regime_name,
            regime_probability=regime_probability,
            timestamp=timestamp,
            reasoning=reason,
        )

    def calculate_position_size(
        self,
        capital: float,
        stop_loss_distance_pct: float,
        max_position_size_pct: float,
    ) -> float:
        risk_per_trade = 0.01
        position_size = min(
            (risk_per_trade / max(stop_loss_distance_pct, 0.001))
            * self.default_leverage,
            max_position_size_pct,
        )
        return position_size


# Backward-compatible aliases per spec
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
BullTrendStrategy = LowVolBullStrategy
MeanReversionStrategy = MidVolCautiousStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Label → strategy class map covering every regime label produced by the HMM.
LABEL_TO_STRATEGY: dict[str, type[BaseRegimeStrategy]] = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": MidVolCautiousStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": MidVolCautiousStrategy,
    "BULL": LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


class StrategyOrchestrator:
    """
    Maps HMM regimes to strategies by VOLATILITY RANKING.

    Regime labels (BEAR/NEUTRAL/BULL) are for display only; strategy selection
    sorts regimes by their expected volatility and buckets into low / mid / high.
    """

    def __init__(self, n_regimes: int = 0):
        self.n_regimes = n_regimes
        self.low_vol_strategy = LowVolBullStrategy()
        self.mid_vol_strategy = MidVolCautiousStrategy()
        self.high_vol_strategy = HighVolDefensiveStrategy()
        self.uncertainty_strategy = HighVolDefensiveStrategy()  # Conservative fallback
        self._cached_strategy_map: dict[int, BaseRegimeStrategy] = {}
        self._last_regime_infos: dict = {}
        self.rebalance_threshold = 0.10  # 10% threshold for rebalancing
        self.last_position_size = 0.0

    def update_regime_infos(self, regime_infos: dict) -> dict[int, BaseRegimeStrategy]:
        """
        Rebuild the volatility-ranked strategy cache.

        Strategy assignment uses volatility rank only:
            rank_fraction = i / max(n - 1, 1)
            <=0.33 → low_vol, <=0.66 → mid_vol, else → high_vol
        """
        if not regime_infos:
            return self._cached_strategy_map

        self._last_regime_infos = dict(regime_infos)
        sorted_by_vol = sorted(
            regime_infos.values(),
            key=lambda r: r.expected_volatility,
        )
        n = len(sorted_by_vol)

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

        logger.info(
            "Updated strategy map for %d regimes: %s", n, self._cached_strategy_map
        )
        return self._cached_strategy_map

    def get_strategy_for_regime(
        self, regime_id: int, regime_infos: dict
    ) -> BaseRegimeStrategy:
        """Select strategy by volatility rank; uncertainty fallback on unknown id."""
        cache_empty = not self._cached_strategy_map
        cache_stale = regime_infos != self._last_regime_infos
        if cache_empty or cache_stale:
            self.update_regime_infos(regime_infos)

        if regime_id in self._cached_strategy_map:
            return self._cached_strategy_map[regime_id]

        logger.warning("Unknown regime_id: %s, using uncertainty strategy", regime_id)
        return self.uncertainty_strategy

    def should_enter_position(
        self,
        strategy: BaseRegimeStrategy,
        *,
        symbol: str,
        current_price: float,
        volatility: float,
        trend: float,
        momentum: float,
        regime_strength: float,
        regime_id: int,
        regime_name: str,
        regime_probability: float,
        confidence: float = 1.0,
        is_flickering: bool = False,
        min_confidence: float = 0.55,
        atr: float = 0.0,
        ema50: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Signal:
        """
        Generate entry signal using the selected strategy.

        Uncertainty mode (confidence < min_confidence OR is_flickering):
            - halves position_size_pct
            - forces leverage to 1.0x
            - appends "[UNCERTAINTY — size halved, leverage clamped]" to reasoning
        """
        signal = strategy.generate_signal(
            symbol=symbol,
            current_price=current_price,
            volatility=volatility,
            trend=trend,
            momentum=momentum,
            regime_strength=regime_strength,
            regime_id=regime_id,
            regime_name=regime_name,
            regime_probability=regime_probability,
            atr=atr,
            ema50=ema50,
            timestamp=timestamp,
        )

        is_uncertain = confidence < min_confidence or is_flickering
        if is_uncertain:
            signal.position_size_pct *= 0.5
            signal.leverage = min(signal.leverage, 1.0)
            signal.reasoning = (
                f"{signal.reasoning} [UNCERTAINTY — size halved, leverage clamped]"
            )
            signal.metadata = {**signal.metadata, "uncertainty": True}

        return signal

    def needs_rebalance(self, target_position_size: float) -> bool:
        """Return True when |target - last| exceeds rebalance_threshold."""
        if target_position_size == 0.0 and self.last_position_size == 0.0:
            return False  # Already flat; skip no-op rebalance
        if self.last_position_size == 0.0:
            return True  # Opening first position or re-entering from flat

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
