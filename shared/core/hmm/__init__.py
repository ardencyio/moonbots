"""HMM Regime Detection Package."""

from .hmm_engine import HMMEngine, RegimeInfo, RegimeState
from .regime_strategies import (
    LABEL_TO_STRATEGY,
    BaseRegimeStrategy,
    Signal,
    StrategyOrchestrator,
    StrategySignal,
    LowVolBullStrategy,
    MidVolCautiousStrategy,
    HighVolDefensiveStrategy,
    # Backward-compatible aliases
    CrashDefensiveStrategy,
    BearTrendStrategy,
    BullTrendStrategy,
    MeanReversionStrategy,
    EuphoriaCautiousStrategy,
)

__all__ = [
    "HMMEngine",
    "RegimeInfo",
    "RegimeState",
    "Signal",
    "StrategySignal",
    "StrategyOrchestrator",
    "BaseRegimeStrategy",
    "LABEL_TO_STRATEGY",
    "LowVolBullStrategy",
    "MidVolCautiousStrategy",
    "HighVolDefensiveStrategy",
    "CrashDefensiveStrategy",
    "BearTrendStrategy",
    "BullTrendStrategy",
    "MeanReversionStrategy",
    "EuphoriaCautiousStrategy",
]
