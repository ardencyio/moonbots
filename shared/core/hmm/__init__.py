"""HMM Regime Detection Package."""

from .hmm_engine import HMMEngine, RegimeInfo, RegimeState
from .regime_strategies import (
    StrategyOrchestrator,
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
    "StrategyOrchestrator",
    "LowVolBullStrategy",
    "MidVolCautiousStrategy",
    "HighVolDefensiveStrategy",
    "CrashDefensiveStrategy",
    "BearTrendStrategy",
    "BullTrendStrategy",
    "MeanReversionStrategy",
    "EuphoriaCautiousStrategy",
]
