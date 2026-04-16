# Moonbots — Trading Bot Farm

Multi-strategy trading bot system with Open Gate strategy for NQ/ES index futures and HMM regime-detection.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check backtest/ shared/ tests/ bots/ scripts/ dashboard/

# Run a bot (paper trading)
uv run python -m bots.nq_open_gate_001.main

# Check bot health
uv run python scripts/bot_status.py

# Emergency stop
uv run python scripts/emergency_stop.py --all

# View P&L report
uv run python scripts/pnl_report.py

# Launch dashboard
uv run streamlit run dashboard/app.py
```

## HMM Regime Backtest

Walk-forward backtest using Hidden Markov Model regime detection. Trains on a warm-up window,
then steps through each out-of-sample bar using `predict_filtered` (no look-ahead bias).

```bash
# SPY daily, 2020-2024 (default)
uv run python scripts/hmm_backtest.py

# Different equities
uv run python scripts/hmm_backtest.py --ticker QQQ --start 2018-01-01 --end 2024-12-31 --interval 1d
uv run python scripts/hmm_backtest.py --ticker AAPL --start 2015-01-01 --end 2024-12-31 --interval 1d

# Crypto (yfinance suffix)
uv run python scripts/hmm_backtest.py --ticker BTC-USD --start 2020-01-01 --end 2024-12-31 --interval 1d
uv run python scripts/hmm_backtest.py --ticker ETH-USD --start 2020-01-01 --end 2024-12-31 --interval 1d

# Futures (yfinance suffix)
uv run python scripts/hmm_backtest.py --ticker NQ=F --start 2018-01-01 --end 2024-12-31 --interval 1d
uv run python scripts/hmm_backtest.py --ticker ES=F --start 2018-01-01 --end 2024-12-31 --interval 1d

# Hourly (limited to ~60 days by yfinance)
uv run python scripts/hmm_backtest.py --ticker SPY --start 2024-10-01 --end 2024-12-31 --interval 1h

# More training data (70% warm-up)
uv run python scripts/hmm_backtest.py --ticker SPY --start 2015-01-01 --end 2024-12-31 --interval 1d --warmup-pct 0.7

# With varlock (injects API keys from 1Password)
varlock run -- uv run python scripts/hmm_backtest.py --ticker SPY --start 2020-01-01 --end 2024-12-31 --interval 1d
```

**Note:** Requires at least ~3 years of daily data (warm-up window + normalisation dropout).
Results are cached as parquet in `data/cache/` — subsequent runs on the same parameters are instant.

## Open Gate Strategy Backtesting

### With yfinance (free, no API key required)

```bash
uv run python -m backtest.examples.yfinance_backtest --source yfinance --days 365
```

The script automatically falls back from 1m → 5m → daily data depending on data availability.

### With Polygon.io API (equities and crypto)

```bash
# Using varlock with 1Password
varlock run -- uv run python -m backtest.examples.polygon_backtest --days 30

# Or with .env file
# 1. Copy env.example to .env and add POLYGON_API_KEY
# 2. Run:
uv run python -m backtest.examples.polygon_backtest --days 30
```

**Supported symbols on Polygon:**

| Type     | Examples                          | Notes                              |
|----------|-----------------------------------|------------------------------------|
| Equities | `SPY`, `QQQ`, `AAPL`, `TSLA`     | Full history on paid tiers         |
| Crypto   | `X:BTCUSD`, `X:ETHUSD`           | Prefix `X:` required               |
| Forex    | `C:EURUSD`, `C:GBPUSD`           | Prefix `C:` required               |

**Note:** Futures (NQ=F, ES=F) are not available via Polygon's stocks API.
Free tier: ~30 days of 5-minute data per month. Daily bars have longer history.

Fetched data is cached as parquet in `data/cache/` — free tier limits are not wasted on repeated fetches.

## Configuration

Copy `env.example` to `.env` and add your API keys:

```bash
cp env.example .env
```

Or use `varlock` with 1Password:

```bash
varlock load   # injects keys from 1Password into the current shell
varlock run -- <command>   # injects keys for a single command
```

## Strategy: Open Gate

The Open Gate strategy detects the first 5-minute candle range at market open (9:30 ET), waits for a breakout through gate_high or gate_low, confirms with a retest and candle pattern (wick rejection or engulfing), then enters with defined stop loss and take profit.

## Strategy: HMM Regime Detection

The HMM strategy uses a Gaussian HMM to classify market regimes (bull/bear/neutral/euphoria).
At each bar it calls `predict_filtered` (forward algorithm) — no future data ever touches the signal.
`StrategyOrchestrator` maps the current regime + volatility bucket to a position size and leverage.

## Adding a New Bot

```bash
uv run python scripts/create_bot.py --id es-momentum-001 --strategy momentum --ticker ES=F
```

## Deployment

```bash
# Build and start bots with Docker
docker compose up -d
```

## Architecture

```
moonbots/
├── backtest/              # Strategy research & backtesting
│   ├── strategies/        # BaseStrategy, OpenGateStrategy
│   ├── data/              # yfinance & Polygon.io fetchers (parquet cache)
│   ├── examples/          # Backtest scripts (yfinance, polygon)
│   ├── backtest.py        # BacktestRunner (metrics, walk-forward, Monte Carlo)
│   ├── optimize.py        # Grid search parameter optimization
│   └── report.py          # Performance report generation
├── shared/core/           # Shared bot infrastructure
│   ├── hmm/               # HMM regime detection
│   │   ├── hmm_engine.py         # GaussianHMM + BIC selection + predict_filtered
│   │   ├── feature_engineering.py # 18-feature OHLCV → HMM input pipeline
│   │   └── regime_strategies.py  # StrategyOrchestrator + volatility buckets
│   ├── risk_manager.py    # RiskGuardrails + RiskManager
│   ├── execution.py       # ExecutionHandler + PaperExecutionHandler
│   ├── state.py           # SQLite-backed StateStore
│   ├── orchestrator.py    # BotOrchestrator + GlobalRiskMonitor
│   ├── metrics.py         # MetricsCollector
│   └── alerts.py          # AlertManager (Discord/Slack webhooks)
├── scripts/               # Operational utilities
│   ├── hmm_backtest.py    # HMM walk-forward backtest CLI
│   ├── bot_status.py      # Health checks
│   ├── pnl_report.py      # P&L reporting
│   ├── emergency_stop.py  # Kill switch
│   ├── reset_daily_stats.py
│   └── create_bot.py      # New bot scaffolding
├── data/cache/            # Parquet cache for fetched OHLCV (gitignored)
├── bots/                  # Per-bot instances
│   └── nq_open_gate_001/  # NQ Open Gate bot
│       ├── bot.json       # Configuration
│       ├── main.py        # Entry point
│       ├── state/         # Bot-specific SQLite state
│       └── logs/          # Log files
├── dashboard/             # Streamlit monitoring app
├── tests/                 # Unit & integration tests
└── Dockerfile.bot         # Bot containerization
```
