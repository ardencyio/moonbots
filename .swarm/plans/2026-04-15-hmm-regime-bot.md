# Plan: HMM Regime-Based Trading Bot Integration

## Approach
Integrate the HMM volatility regime trading system into the existing moonbots architecture. Rather than duplicating infrastructure, we'll add HMM-specific components to `shared/core/hmm/` and create a new bot instance in `bots/hmm_spy_001/`. The most critical piece is implementing the **forward algorithm** (not Viterbi) to prevent look-ahead bias—the core differentiator of the HMM prompt's design.

## Stories
Ordered list, each completable in one iteration:

### Story 1: HMM Core Engine (No Look-Ahead Forward Algorithm)
**US-001**: Implement `shared/core/hmm/hmm_engine.py` with GaussianHMM, BIC model selection, and **forward algorithm** for filtered inference (using only past+present data).  
**Why first**: This is the most critical component per MULTI.md. Must validate no look-ahead bias before anything else.

### Story 2: Feature Engineering Pipeline  
**US-002**: Implement `shared/core/hmm/feature_engineering.py` with OHLCV → technical features (returns, volatility, RSI, ADX, ATR, etc.), rolling z-score normalization.  
**Depends on US-001**: HMM needs features to train/predict.

### Story 3: Volatility-Based Strategy Classes  
**US-003**: Implement `shared/core/hmm/regime_strategies.py` with LowVol/MidVol/HighVol strategy classes, StrategyOrchestrator mapping regimes by volatility rank (not labels).  
**Depends on US-001, US-002**: Requires HMM regime detection.

### Story 4: Backtest Integration  
**US-004**: Create `backtest/strategies/hmm_regime_strategy.py` extending BaseStrategy, integrate with existing BacktestRunner.  
**Depends on US-003**: Leverages strategy classes for backtesting.

### Story 5: Look-Ahead Bias Test  
**US-005**: Implement `tests/hmm/test_look_ahead.py` verifying regime at T is identical with data[0:T] vs data[0:T+100].  
**Depends on US-001, US-002**: Critical validation before live use.

### Story 6: Bot Instance Scaffolding  
**US-006**: Create `bots/hmm_spy_001/` directory with bot.json config, main.py entry point wiring shared components.  
**Depends on US-004**: Only scaffold bot after backtests validate logic.

### Story 7: Alpaca Live Data Feed  
**US-007**: Extend `backtest/data/fetcher.py` or create `shared/data/` module for live Alpaca WebSocket bars (historical fetch + real-time subscription).  
**Depends on US-006**: Bot needs live data.

### Story 8: Risk Manager Enhancements  
**US-008**: Extend `shared/core/risk_manager.py` with position sizing by uncertainty mode, regime-aware max leverage, gap risk calculations.  
**Depends on US-003**: Aligns with HMM strategy requirements.

### Story 9: Dashboard Integration  
**US-009**: Extend `dashboard/app.py` to display HMM regime info (current regime, probability, stability, flicker rate).  
**Depends on US-006**: Bot instance needs monitoring.

### Story 10: Configuration & Documentation  
**US-010**: Update settings examples, add HMM params to bot.json schema, document HMM architecture in README.  
**Depends on all**: Final polish.

## Files to Create

### New Files
- `shared/core/hmm/__init__.py` — package init exposing HMMEngine, StrategyOrchestrator
- `shared/core/hmm/hmm_engine.py` — GaussianHMM with BIC selection, forward algorithm (filtered inference)
- `shared/core/hmm/regime_strategies.py` — LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy, StrategyOrchestrator
- `shared/core/hmm/feature_engineering.py` — OHLCV feature computation with rolling normalization
- `backtest/strategies/hmm_regime_strategy.py` — HMM strategy wrapper for BacktestRunner
- `tests/hmm/__init__.py` — test package init
- `tests/hmm/test_hmm_engine.py` — HMM unit tests
- `tests/hmm/test_look_ahead.py` — critical look-ahead bias validation test
- `tests/hmm/test_hmm_strategy.py` — strategy tests
- `bots/hmm_spy_001/bot.json` — bot configuration with symbols, HMM params, thresholds
- `bots/hmm_spy_001/main.py` — thin entry point wiring shared components
- `bots/hmm_spy_001/__init__.py` — package marker

### Modified Files
- `shared/core/risk_manager.py` — add uncertainty mode sizing, gap risk, regime-aware leverage limits
- `backtest/strategies/__init__.py` — export HMMRegimeStrategy
- `backtest/backtesting_runner.py` — ensure HMM strategy is compatible (may need minimal changes)
- `dashboard/app.py` — add HMM-specific metrics display
- `README.md` — document HMM architecture

## Risks

1. **Look-ahead bias in HMM**: Using `model.predict()` runs Viterbi which revises past states with future data. **Mitigation**: Implement forward algorithm manually as specified in HMM.md Phase 2, require test_look_ahead.py to pass before proceeding.

2. **Regime misclassification in volatile markets**: **Mitigation**: Uncertainty mode with reduced position sizes, flicker detection threshold.

3. **Alpaca API rate limits**: **Mitigation**: Exponential backoff retry, batch historical requests.

4. **Rebalance churn**: Too many regime transitions causing excessive trades. **Mitigation**: 10% rebalance threshold, 3-bar persistence filter.

5. **Integration complexity with existing bot (NQ Open Gate)**: **Mitigation**: Keep HMM components isolated in `shared/core/hmm/`, only affect backtest runner through strategy interface.

## Out of Scope

- Short selling (HMM bot is ALWAYS LONG per design)
- Multiple bot instances beyond hmm_spy_001 for now (can duplicate later)
- Email/webhook alerts (existing alerts.py can be extended later)
- Complex sector correlation checks (single-asset focus initially)
- Monte Carlo optimization (use HMM's BIC-based model selection)

## Validation Criteria
- `test_look_ahead.py` passes (regime identical with truncated data)
- Walk-forward backtest shows positive Sharpe on 2019-2024 SPY data
- Paper trade for 1 week with no critical errors
- No look-ahead bias confirmed by comparing truncated vs full backtests
