# Claude Code Prompt: HMM Regime-Based Trading Bot — Final

**How to use:** Copy each phase into Claude Code one at a time. Complete and test each phase before moving to the next.

---

## PHASE 1: Project Scaffolding & Environment Setup

Create a Python project called "regime-trader" with the following structure:

```
regime-trader/
├── config/
│   ├── settings.yaml        # All configurable parameters
│   └── credentials.yaml.example
├── core/
│   ├── __init__.py
│   ├── hmm_engine.py        # HMM regime detection engine
│   ├── regime_strategies.py # Vol-based allocation strategies
│   ├── risk_manager.py      # Position sizing, leverage, drawdown limits
│   └── signal_generator.py  # Combines HMM + strategy into signals
├── broker/
│   ├── __init__.py
│   ├── alpaca_client.py     # Alpaca API wrapper
│   ├── order_executor.py    # Order placement, modification, cancellation
│   └── position_tracker.py  # Track open positions, P&L
├── data/
│   ├── __init__.py
│   ├── market_data.py       # Real-time and historical data fetching
│   └── feature_engineering.py # Technical indicators, feature computation
├── monitoring/
│   ├── __init__.py
│   ├── logger.py            # Structured logging
│   ├── dashboard.py         # Terminal-based live dashboard
│   └── alerts.py            # Email/webhook alerts for critical events
├── backtest/
│   ├── __init__.py
│   ├── backtester.py        # Walk-forward allocation backtester
│   ├── performance.py       # Sharpe, drawdown, regime breakdown, benchmarks
│   └── stress_test.py       # Crash injection, gap simulation
├── tests/
│   ├── test_hmm.py
│   ├── test_look_ahead.py   # Verify no look-ahead bias
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_orders.py
├── main.py                  # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

Set up requirements.txt with:
- hmmlearn
- alpaca-trade-api
- alpaca-py
- pandas, numpy, scipy
- ta (technical analysis library)
- scikit-learn
- pyyaml
- python-dotenv
- websocket-client
- schedule
- rich (for terminal dashboard)

Create settings.yaml with ALL parameters, grouped by section, with defaults and comments:
- broker (paper_trading: true, symbols: [SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, AMD], timeframe: 1Day)
- hmm (n_candidates: [3, 4, 5, 6, 7], n_init: 10, covariance_type: full, min_train_bars: 252, stability_bars: 3, flicker_window: 20, flicker_threshold: 4, min_confidence: 0.55)
- strategy (low_vol_allocation: 0.95, mid_vol_allocation_trend: 0.95, mid_vol_allocation_no_trend: 0.60, high_vol_allocation: 0.60, low_vol_leverage: 1.25, rebalance_threshold: 0.10, uncertainty_size_mult: 0.50)
- risk (max_risk_per_trade: 0.01, max_exposure: 0.80, max_leverage: 1.25, max_single_position: 0.15, max_concurrent: 5, max_daily_trades: 20, daily_dd_reduce: 0.02, daily_dd_halt: 0.03, weekly_dd_reduce: 0.05, weekly_dd_halt: 0.07, max_dd_from_peak: 0.10)
- backtest (slippage_pct: 0.0005, initial_capital: 100000, train_window: 252, test_window: 126, step_size: 126, risk_free_rate: 0.045)
- monitoring (dashboard_refresh_seconds: 5, alert_rate_limit_minutes: 15)

Create .env.example with:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

Do NOT implement any logic yet — just the skeleton with imports, class stubs, type hints, and docstrings.
Add .env and credentials.yaml to .gitignore.

---

## PHASE 2: HMM Regime Detection Engine

Implement core/hmm_engine.py and data/feature_engineering.py.

**DESIGN PHILOSOPHY:** The HMM is a VOLATILITY CLASSIFIER. It detects whether the market is in a calm, moderate, or turbulent volatility environment. It does NOT predict price direction. The strategy layer uses the volatility classification to set portfolio allocation — be fully invested when conditions are calm, reduce when turbulent.

### REQUIREMENTS:

#### 1. GAUSSIAN HMM WITH AUTOMATIC MODEL SELECTION:
- Test n_components = [3, 4, 5, 6, 7] during training
- For each candidate, train and compute BIC (Bayesian Information Criterion)
- BIC = -2 \* log_likelihood + n_params \* log(n_samples)
- Select lowest BIC score (simplest model that explains the data)
- Run multiple random initializations per candidate (n_init=10)
- Log ALL candidate BIC scores and which was selected

After training, sort regimes by mean return (ascending) for LABELING:
- Lowest return → CRASH / BEAR
- Highest return → BULL / EUPHORIA
- Assign labels based on selected count:
  - 3 regimes: BEAR, NEUTRAL, BULL
  - 4 regimes: CRASH, BEAR, BULL, EUPHORIA
  - 5 regimes: CRASH, BEAR, NEUTRAL, BULL, EUPHORIA
  - 6 regimes: CRASH, STRONG\_BEAR, WEAK\_BEAR, WEAK\_BULL, STRONG\_BULL, EUPHORIA
  - 7 regimes: CRASH, STRONG\_BEAR, WEAK\_BEAR, NEUTRAL, WEAK\_BULL, STRONG\_BULL, EUPHORIA

IMPORTANT: Labels are sorted by return for human readability. But the STRATEGY layer sorts by VOLATILITY independently. The labels don't drive strategy decisions.

#### 2. OBSERVABLE FEATURES (inputs to HMM):
Implement in data/feature_engineering.py as pure functions.

Compute from OHLCV:
- Returns: log returns over 1, 5, 20 periods
- Volatility: realized vol (20-period rolling std), vol ratio (5-period / 20-period)
- Volume: normalized volume (z-score vs 50-period mean), volume trend (slope of 10-period SMA)
- Trend: ADX (14-period), slope of 50-period SMA
- Mean reversion: RSI(14) z-score, distance from 200 SMA as % of price
- Momentum: ROC 10 and 20 period
- Range: normalized ATR (14-period ATR / close)

Standardize ALL features with rolling z-scores (252-period lookback).

#### 3. MODEL TRAINING:
- hmmlearn.GaussianHMM, covariance\_type="full"
- Minimum 2 years daily data (504 trading days)
- Expanding window retraining: retrain at configurable intervals
- Store model with pickle + metadata (n\_regimes, bic, training\_date, labels)
- Log: likelihood, BIC, convergence, iterations

#### 4. REGIME DETECTION — NO LOOK-AHEAD BIAS:

**\*\*\* THIS IS THE MOST IMPORTANT TECHNICAL DETAIL. \*\*\***

DO NOT use model.predict(). predict() runs the Viterbi algorithm which processes the ENTIRE sequence and revises past states using future data. This is look-ahead bias that makes backtests unrealistically good.

INSTEAD implement FORWARD ALGORITHM ONLY (filtered inference):

```python
def predict_regime_filtered(self, features_up_to_now):
    """
    Compute P(state_t | observations_1:t) using forward algorithm.
    Uses ONLY past and present data. No future data.
    """
    # Use model's startprob_, transmat_, means_, covars_
    # Implement forward pass manually:
    # 1. alpha_0 = startprob * emission_prob(obs_0)
    # 2. alpha_t = (alpha_{t-1} @ transmat) * emission_prob(obs_t)
    # 3. Normalize at each step (work in log space)
    # 4. alpha_T = filtered distribution at current time
    # Cache previous alpha for efficiency in live/backtest loop
```

MANDATORY TEST — tests/test\_look\_ahead.py:

```python
def test_no_look_ahead_bias():
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    model = train_hmm(full_data)
    regime_short = predict_regime_filtered(data[0:400])[-1]
    regime_long = predict_regime_filtered(data[0:500])[400]
    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"
```

#### 5. REGIME STABILITY FILTER:
- Regime change only "confirmed" after persisting N bars (default 3)
- During transition: keep previous regime, reduce sizes by 25%
- Track flicker rate (changes per 20 bars)
- If flicker rate > threshold (default 4): force uncertainty mode

#### 6. ADDITIONAL METHODS:
- predict\_regime\_proba() -> probability distribution
- get\_regime\_stability() -> consecutive bars in current regime
- get\_transition\_matrix() -> learned transition probabilities
- detect\_regime\_change() -> True only if confirmed
- get\_regime\_flicker\_rate() -> changes per window
- is\_flickering() -> True if flicker rate exceeds threshold

#### 7. REGIME METADATA:

RegimeInfo dataclass:
- regime\_id, regime\_name, expected\_return, expected\_volatility
- recommended\_strategy\_type, max\_leverage\_allowed
- max\_position\_size\_pct, min\_confidence\_to\_act

RegimeState dataclass:
- label, state\_id, probability, state\_probabilities
- timestamp, is\_confirmed, consecutive\_bars

Log regime changes as WARNING. Log confirmations as INFO.

---

## PHASE 3: Volatility-Based Allocation Strategy

Implement core/regime\_strategies.py — the allocation layer that sizes positions based on the HMM's volatility regime detection.

**DESIGN INSIGHT:** The HMM excels at detecting VOLATILITY ENVIRONMENTS, not market direction. Stocks trend upward roughly 70% of the time in low-volatility periods. The worst drawdowns cluster in high-volatility spikes. So the strategy is simple:

- Low vol → be fully invested (calm markets trend up)
- Mid vol → stay invested if trend intact, reduce if not
- High vol → reduce but stay partially invested (catch V-shaped rebounds)

The edge comes from AVOIDING BIG DRAWDOWNS through vol-based sizing. When you cut your worst drawdown in half, compounding works in your favor over time.

**ALWAYS LONG. NEVER SHORT.**
Shorting was tested extensively in walk-forward backtesting and consistently destroyed returns because:
1. Markets have long-term upward drift
2. V-shaped recoveries happen fast and the HMM is 2-3 days late detecting them
3. Short positions during rebounds wipe out crash gains

The correct response to high volatility is REDUCING allocation, not reversing direction.

### THREE STRATEGY CLASSES (based on volatility rank):

**1. LowVolBullStrategy** (lowest third of regimes by expected\_volatility):
- Direction: LONG
- Allocation: 95% of portfolio
- Leverage: 1.25x (modest leverage in calm conditions)
- Stop: max(price - 3 ATR, 50 EMA - 0.5 ATR)
- This is where most returns are generated. Calm markets + leverage = compounding.

**2. MidVolCautiousStrategy** (middle third by expected\_volatility):
- Direction: LONG
- If price > 50 EMA: allocation 95%, leverage 1.0x (trend intact, stay invested)
- If price < 50 EMA: allocation 60%, leverage 1.0x (trend broken, reduce)
- Stop: 50 EMA - 0.5 ATR

**3. HighVolDefensiveStrategy** (top third by expected\_volatility):
- Direction: LONG (NOT short)
- Allocation: 60% of portfolio
- Leverage: 1.0x
- Stop: 50 EMA - 1.0 ATR (wider for volatile conditions)
- Staying 60% invested catches the sharp rebounds after selloffs.

### VOLATILITY RANK MAPPING:
For any regime count (3-7), map each regime's vol rank to a strategy:
```
position = rank / (n_regimes - 1)  # 0.0 = lowest vol, 1.0 = highest
position <= 0.33 → LowVolBullStrategy
position >= 0.67 → HighVolDefensiveStrategy
else → MidVolCautiousStrategy
```

### STRATEGY ORCHESTRATOR:
- Takes regime\_infos from HMM
- Sorts by expected\_volatility (ascending) to compute vol\_rank per regime
- Maps regime\_id → vol\_rank → strategy class
- This sort is INDEPENDENT of the label sort (which is by return)
- "BULL" label does NOT mean low vol. The orchestrator ignores labels.

### CONFIDENCE AND UNCERTAINTY:
- Minimum confidence threshold: 0.55 (configurable)
- Uncertainty mode triggers when: prob < threshold, or is\_flickering=True
- In uncertainty mode: halve all position sizes, force leverage to 1.0x
- Append "[UNCERTAINTY — size halved]" to reasoning

### REBALANCING:
- Only rebalance when target allocation differs from current by >10%
- This prevents churn from minor probability fluctuations
- Fewer trades = less slippage = better real-world performance

### IMPLEMENTATION:
- BaseStrategy ABC: generate\_signal(symbol, bars, regime\_state) -> Optional[Signal]
- LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy
- StrategyOrchestrator:
  - \_\_init\_\_(config, regime\_infos): sorts by vol, maps strategies
  - generate\_signals(symbols, bars, regime\_state, is\_flickering) -> list[Signal]
  - update\_regime\_infos(regime\_infos): rebuilds mapping after HMM retrain

Signal dataclass:
- symbol, direction (LONG or FLAT), confidence, entry\_price, stop\_loss
- take\_profit (Optional), position\_size\_pct (0.60 to 0.95)
- leverage (1.0 or 1.25), regime\_id, regime\_name, regime\_probability
- timestamp, reasoning, strategy\_name, metadata

Keep backward-compatible aliases:
- CrashDefensiveStrategy = HighVolDefensiveStrategy
- BearTrendStrategy = HighVolDefensiveStrategy
- MeanReversionStrategy = MidVolCautiousStrategy
- BullTrendStrategy = LowVolBullStrategy
- EuphoriaCautiousStrategy = LowVolBullStrategy
- etc.
- LABEL\_TO\_STRATEGY dict for all possible labels → strategy class

---

## PHASE 4: Walk-Forward Backtesting & Validation

Implement backtest/backtester.py, backtest/performance.py, and backtest/stress\_test.py

This is an ALLOCATION-BASED walk-forward backtester. It does NOT track individual trade entries and exits. It sets a target portfolio allocation each bar based on the detected volatility regime and rebalances when the allocation changes meaningfully. This is how real systematic strategies work.

### 1. WALK-FORWARD OPTIMIZATION ENGINE (backtester.py):

Rolling windows:
- In-Sample (IS): 252 trading days (1 year) for HMM training + model selection
- Out-of-Sample (OOS): 126 trading days (6 months) for evaluation
- Step size: 126 trading days (6 months)

For each window:
a. Train HMM on IS data (BIC model selection)
b. Compute vol rankings from trained model's regime\_infos
c. Walk through OOS bar by bar:
   - Compute features using ONLY data up to current bar
   - Run filtered HMM (forward algorithm)
   - Get strategy signal: target allocation based on vol rank
   - If allocation changed >10% from current → rebalance
   - Mark to market: equity = cash + shares \* price
d. Record regime predictions and equity at each bar
e. Record a "trade" whenever allocation changes for metrics

ALLOCATION MATH (must be exactly correct):
```
equity = cash + shares * current_price
target_shares = int(equity * target_allocation / current_price)
delta = target_shares - current_shares
cash -= delta * current_price
shares = target_shares
```

When leverage > 1.0 (e.g., 1.25x in low vol), target\_allocation > 1.0, so target\_shares \* price > equity, making cash negative. This is margin. equity = cash + shares \* price is still correct because share value exceeds the margin debt. Alpaca supports this with 2x overnight leverage.

REALISTIC SIMULATION:
- Slippage: 0.05% on each rebalance (configurable)
- Rebalancing threshold: 10% (prevents churn)
- Fill delay: 1 bar (signal bar N → rebalance at bar N+1 open)
- No individual trade stops in backtester (stops are for live trading only)
- Commission: $0 default (Alpaca commission-free)

### 2. PERFORMANCE METRICS (performance.py):

Core:
- Total return (%), CAGR
- Sharpe ratio (annualized), Sortino ratio
- Calmar ratio (CAGR / max drawdown)
- Max drawdown: percentage AND duration in trading days
- Win rate, avg win/loss, profit factor
- Total trades, avg holding period

Regime-specific (table format):
Regime | % Time In | Return Contribution | Avg Trade P&L | Win Rate | Sharpe
Show for each detected regime. This proves each vol environment's strategy is performing as expected.

Confidence-bucketed:
Confidence | Trades | Sharpe | Win Rate | Avg P&L
< 50%, 50-60%, 60-70%, 70%+
If high-confidence trades outperform low-confidence → HMM adds value.

Benchmark comparisons (run automatically with --compare flag):
a. Buy-and-hold: hold the asset entire period
b. 200 SMA trend: long above 200 SMA, cash below
c. Random entry + same risk management: random allocation changes at same frequency, same position sizing rules. 100 random seeds, report mean/std.

Worst-case: worst day, worst week, worst month, max consecutive losses, longest time underwater.

Output:
- Rich formatted tables to terminal
- equity\_curve.csv, trade\_log.csv, regime\_history.csv, benchmark\_comparison.csv

### 3. STRESS TESTING (stress\_test.py):

a. Crash injection: insert -5% to -15% single-day gaps at 10 random points. Run 100 Monte Carlo simulations. Report: mean max loss, worst case, % where circuit breaker fired.

b. Gap risk: insert overnight gaps of 2-5x ATR at random points. Report: expected loss vs actual.

c. Regime misclassification: deliberately shuffle regime labels. Verify risk management contains damage even with wrong regimes. If system blows up → risk management isn't independent enough.

```
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare
python main.py backtest --stress-test
```

---

## PHASE 5: Risk Management Layer

Implement core/risk\_manager.py.

The risk manager operates INDEPENDENTLY of the HMM. Even if the HMM fails completely, circuit breakers catch drawdowns based on actual P&L. Defense in depth. The risk manager has ABSOLUTE VETO POWER over any signal.

### 1. PORTFOLIO-LEVEL LIMITS:
- Max total exposure: 80% of portfolio (20% cash minimum — note: when using 1.25x leverage, the notional exposure exceeds equity but the margin requirement stays within Alpaca's limits)
- Max single position: 15%
- Max correlated exposure: 30% in one sector
- Max concurrent positions: 5
- Max daily trades: 20
- Max portfolio leverage: 1.25x

### 2. CIRCUIT BREAKERS (fire on actual P&L, independent of regime):
- Daily DD > 2%: reduce all sizes 50% rest of day
- Daily DD > 3%: close ALL positions, halt rest of day
- Weekly DD > 5%: reduce all sizes 50% rest of week
- Weekly DD > 7%: close ALL, halt rest of week
- Peak DD > 10%: halt ALL trading, write trading\_halted.lock file requiring manual deletion to resume

Log every trigger with: breaker type, actual DD, equity, positions closed, HMM regime at time (track if HMM was wrong).

### 3. POSITION-LEVEL RISK:
- Every position MUST have a stop loss — system refuses orders without one
- Max risk per trade: 1% of portfolio
- Position size = (portfolio \* 0.01) / abs(entry - stop\_loss)
- Cap at regime max, then portfolio max (15%)
- Minimum position: $100
- GAP RISK: overnight positions assume 3x stop gap-through. Overnight size = min(normal, size where 3x gap = 2% of portfolio)

### 4. LEVERAGE RULES:
- Default: 1.0x
- Only low-vol regimes may use up to 1.25x
- Force 1.0x if: regime uncertain, any circuit breaker active, 3+ positions open, high flicker rate
- Alpaca supports 2x overnight (Reg T, $2k+ equity) and 4x intraday (PDT, $25k+ equity). Our 1.25x max is deliberately conservative.

### 5. ORDER VALIDATION:
- Check buying power, tradeable status, bid-ask spread < 0.5%
- Block duplicates (same symbol + direction within 60 seconds)
- Log every rejection with structured reason

### 6. CORRELATION CHECK:
- 60-day rolling correlation with existing positions
- Correlation > 0.7: reduce size 50%
- Correlation > 0.85: reject trade

### IMPLEMENTATION:
- RiskManager: validate\_signal(signal, portfolio\_state) -> RiskDecision
- RiskDecision: approved, modified\_signal, rejection\_reason, modifications list
- PortfolioState: equity, cash, buying\_power, positions, daily/weekly pnl, peak equity, drawdown, circuit\_breaker\_status, flicker\_rate
- CircuitBreaker: check(), update(pnl), reset\_daily(), reset\_weekly(), get\_history()
- All thresholds from settings.yaml

---

## PHASE 6: Alpaca Broker Integration

Implement the broker/ package.

### 1. broker/alpaca\_client.py:
- alpaca-py SDK wrapper
- Credentials from .env (NEVER hardcoded, .env in .gitignore)
- Paper: https://paper-api.alpaca.markets (DEFAULT)
- Live: https://api.alpaca.markets
- If paper\_trading: false, require confirmation:
  "⚠️ LIVE TRADING MODE. Type 'YES I UNDERSTAND THE RISKS' to confirm:"
- Methods: get\_account(), get\_positions(), get\_order\_history(), is\_market\_open(), get\_clock(), get\_available\_margin()
- Health check on startup, auto-reconnect with exponential backoff

### 2. broker/order\_executor.py:
- submit\_order(signal): LIMIT orders by default (+/- 0.1% of current price), cancel after 30s if unfilled, optionally retry at market
- submit\_bracket\_order(signal): entry + stop + take\_profit via Alpaca OCO
- modify\_stop(symbol, new\_stop): only tighten, never widen
- cancel\_order(), close\_position(), close\_all\_positions()
- Unique trade\_id linking signal → risk\_decision → order → fill

### 3. broker/position\_tracker.py:
- WebSocket subscription for instant fill notifications
- Update PortfolioState and CircuitBreaker on every fill
- Per-position tracking: entry time/price, current price, unrealized P&L, stop level, holding period, regime at entry vs current
- Sync with Alpaca on startup (reconcile tracked vs actual positions)

### 4. data/market\_data.py:
- get\_historical\_bars(symbol, timeframe, start, end)
- subscribe\_bars(symbols, timeframe, callback) via WebSocket
- subscribe\_quotes(symbols, callback) for spread checks
- get\_latest\_bar(), get\_latest\_quote(), get\_snapshot()
- Handle gaps (weekends, holidays, halts) gracefully

---

## PHASE 7: Main Loop & Orchestration

Implement main.py.

### STARTUP:
1. Load config, connect to Alpaca, verify account
2. Check market hours (wait or exit if closed)
3. Load or train HMM (if model >7 days old or missing, retrain)
4. Initialize risk manager with current portfolio from Alpaca
5. Initialize position tracker, sync positions
6. Check for state\_snapshot.json (recovery from previous session)
7. Start WebSocket data feeds
8. Print system state, log "System online"

### MAIN LOOP (each bar close, default 5-min bars):
1. New bar from WebSocket
2. Compute features (rolling window, no future data)
3. Filtered HMM prediction (forward algorithm only)
4. Regime stability check (3-bar persistence)
5. Flicker rate check → uncertainty mode if high
6. StrategyOrchestrator: target allocation per symbol
7. For each signal: risk\_manager.validate\_signal()
   - → approved: order\_executor.submit\_order()
   - → modified: log, submit modified
   - → rejected: log reason
8. Update trailing stops per regime
9. Circuit breaker check
10. Dashboard refresh
11. Weekly: retrain HMM

### SHUTDOWN (SIGINT/SIGTERM):
- Close WebSocket connections
- Do NOT close positions (stops in place)
- Save state\_snapshot.json
- Print session summary

### ERROR HANDLING:
- Alpaca API: 3 retries, exponential backoff
- HMM error: hold current regime
- Data feed drop: pause signals, keep stops active
- Unhandled: log traceback, save state, alert

### CLI:
```
--dry-run      Full pipeline, no orders
--backtest     Walk-forward backtester
--train-only   Train HMM and exit
--stress-test  Run stress tests
--compare      Benchmark comparisons
--dashboard    Show dashboard for running instance
```

---

## PHASE 8: Monitoring, Alerts & Dashboard

Implement monitoring/ package.

### 1. monitoring/logger.py:
- Structured JSON logging
- Rotating files (10MB, 30 days): main.log, trades.log, alerts.log, regime.log
- Every entry includes: timestamp, regime, probability, equity, positions, daily\_pnl

### 2. monitoring/dashboard.py (rich library):
```
┌─ REGIME ──────────────────────────────────────────────────┐
│ BULL (72%) │ Stability: 14 bars │ Flicker: 1/20           │
├─ PORTFOLIO ────────────────────────────────────────────────┤
│ Equity: $105,230 │ Daily: +$340 (+0.32%)                   │
│ Allocation: 95% │ Leverage: 1.25x                          │
├─ POSITIONS ────────────────────────────────────────────────┤
│ SPY │ LONG │ $520.30 │ +1.2% │ Stop: $508 │ 3h            │
├─ RECENT SIGNALS ───────────────────────────────────────────┤
│ 14:30 │ SPY │ Rebalance 60%→95% │ Low vol                  │
├─ RISK STATUS ──────────────────────────────────────────────┤
│ Daily DD: 0.3%/3% ✅ │ From Peak: 1.2%/10% ✅             │
├─ SYSTEM ───────────────────────────────────────────────────┤
│ Data: ✅ │ API: ✅ 23ms │ HMM: 2d ago │ PAPER             │
└───────────────────────────────────────────────────────────┘
```

Refresh every 5 seconds. Color-coded risk bars.

### 3. monitoring/alerts.py:
- Triggers: regime change, circuit breaker, large P&L, data feed down, API lost, HMM retrained, flicker exceeded
- Delivery: console, log file, email (optional), webhook (optional)
- Rate limit: 1 per event type per 15 minutes

---

## PHASE 9: Integration Testing & Documentation

### 1. TESTS:
a. End-to-end dry run: data → HMM → strategy → risk → simulated orders
b. Look-ahead bias: test\_look\_ahead.py passes, backtest identical with different end dates
c. Risk stress: extreme signals capped, rapid-fire blocked, no-stop rejected
d. Alpaca paper: place bracket order, modify stop, cancel, verify clean state
e. Recovery: kill process, restart, verify state recovery and no double-entry

### 2. README.md:
- Philosophy: "risk management > signal generation"
- Architecture diagram: data → features → HMM → vol rank → allocation → risk → broker
- Quick start (6 steps)
- CLI reference
- Configuration guide
- FAQ: forward algorithm, BIC selection, trade rejections, live trading switch
- Disclaimer: educational, no guaranteed profits, paper trade first

### 3. settings.yaml: complete, documented, production-ready defaults for paper trading
