Great question. Looking at both structures, the good news is that **moonbots is already well-architected for multi-bot integration** — you don't need to restructure it significantly. Here's my honest assessment:

## You Don't Need a New Repo

Your existing moonbots repo already has the right mental model: shared infrastructure + per-bot instances. The HMM bot fits naturally as another entry under `bots/`.

## What Maps Directly (Reuse As-Is)

Your existing moonbots components that the HMM bot can plug straight into:

- `shared/core/risk_manager.py` → replaces the HMM prompt's `core/risk_manager.py` (yours already handles circuit breakers, position limits, etc.)
- `shared/core/execution.py` → replaces `broker/order_executor.py`
- `shared/core/orchestrator.py` → replaces `main.py` loop logic
- `shared/core/alerts.py` → replaces `monitoring/alerts.py`
- `shared/core/state.py` → replaces `state_snapshot.json` pattern
- `dashboard/` → replaces `monitoring/dashboard.py`
- `backtest/` → replaces the entire `backtest/` module from the HMM prompt (your BacktestRunner already does walk-forward + Monte Carlo)

## What Needs to Be Added

The genuinely new HMM-specific code that moonbots doesn't have:

```
moonbots/
├── shared/
│   └── core/
│       └── hmm/                     # NEW — shared, reusable HMM engine
│           ├── __init__.py
│           ├── hmm_engine.py        # GaussianHMM + BIC selection + forward algo
│           ├── regime_strategies.py # LowVol/MidVol/HighVol strategy classes
│           └── feature_engineering.py  # OHLCV → features pipeline
│
├── backtest/
│   └── strategies/
│       └── hmm_regime_strategy.py   # NEW — HMM strategy for BacktestRunner
│
├── bots/
│   └── hmm_spy_001/                 # NEW — first HMM bot instance
│       ├── bot.json                 # Config (symbols, HMM params, thresholds)
│       ├── main.py                  # Thin entry point, wires shared/ components
│       ├── state/
│       └── logs/
│
└── tests/
    └── hmm/                         # NEW
        ├── test_hmm_engine.py
        ├── test_look_ahead.py       # Critical — forward algo bias test
        └── test_hmm_strategy.py
```

## How the Wiring Works

The HMM bot's `main.py` becomes a thin orchestration file that wires together shared components:

```python
# bots/hmm_spy_001/main.py
from shared.core.orchestrator import BotOrchestrator
from shared.core.hmm.hmm_engine import HMMEngine
from shared.core.hmm.regime_strategies import StrategyOrchestrator

# HMM-specific logic lives in shared/core/hmm/
# Everything else (risk, execution, state, alerts) is reused from shared/core/
```

## Key Integration Decisions to Make

**1. Broker abstraction** — Your `shared/core/execution.py` presumably wraps a specific broker. The HMM prompt is Alpaca-specific. You'll want to make sure your ExecutionHandler either already supports Alpaca or you add an `AlpacaExecutionHandler` alongside whatever you're currently using.

**2. Data fetching** — Your `backtest/data/` uses yfinance and Polygon. The HMM bot needs live bars via WebSocket (Alpaca). You'll want a `shared/data/` module (or expand `backtest/data/`) with a `market_data.py` that handles both historical (backtesting) and live WebSocket feeds.

**3. Strategy interface** — Check whether your existing `BaseStrategy` in `backtest/strategies/` is compatible with what the HMM strategy needs. If it expects a `generate_signal(symbol, bars, regime_state)` signature, you may need to extend the base class rather than replace it.

**4. bot.json schema** — You'll need HMM-specific config keys (n_candidates, flicker_threshold, min_confidence, etc.) alongside your existing bot config fields. A clean approach is a versioned schema with a `strategy_type` field that gates which config keys are expected.

## Recommended Approach: One Repo, Incremental

The practical path is:

1. Add `shared/core/hmm/` with the three core HMM files
2. Add `hmm_regime_strategy.py` to `backtest/strategies/` so you can backtest it with your existing BacktestRunner
3. Run backtests and validate performance before any live wiring
4. Only then scaffold `bots/hmm_spy_001/` and wire to live execution

This way you validate the HMM logic independently before integrating it into the live orchestration, and your existing NQ Open Gate bot is completely unaffected throughout.

## One Thing Worth Auditing First

The most critical piece from the HMM prompt — the **forward algorithm (no look-ahead bias)** — is entirely self-contained in `hmm_engine.py` and has no dependencies on any other moonbots component. That's the first thing to implement and test in isolation before anything else.
