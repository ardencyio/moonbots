# Moonbots — Trading Bot Farm

Multi-strategy trading bot system with Open Gate strategy for NQ/ES index futures.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check backtest/ shared/ tests/ bots/ scripts/ dashboard/

# Run a bot (paper trading)
uv run python bots/nq_open_gate_001/main.py

# Check bot health
uv run python scripts/bot_status.py

# Emergency stop
uv run python scripts/emergency_stop.py --all

# View P&L report
uv run python scripts/pnl_report.py

# Launch dashboard
uv run streamlit run dashboard/app.py
```

## Backtesting

### With yfinance (free, no API key required)

```bash
uv run python -m backtest.examples.yfinance_backtest --source yfinance --days 365
```

The script automatically falls back from 1m → 5m → daily data depending on data availability.

### With Polygon.io API (requires API key)

```bash
# Using varlock with 1Password
varlock run -- uv run python -m backtest.examples.polygon_backtest --days 30

# Or with .env file
# 1. Copy env.example to .env and add POLYGON_API_KEY
# 2. Run:
uv run python -m backtest.examples.polygon_backtest --days 30
```

**Note:** Polygon's free tier limits 5-minute data to ~30 days per month.

## Architecture

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check backtest/ shared/ tests/ bots/ scripts/ dashboard/

# Run a bot (paper trading)
uv run python bots/nq_open_gate_001/main.py

# Check bot health
uv run python scripts/bot_status.py

# Emergency stop
uv run python scripts/emergency_stop.py --all

# View P&L report
uv run python scripts/pnl_report.py

# Launch dashboard
uv run streamlit run dashboard/app.py
```

## Architecture

```
moonbots/
├── backtest/              # Strategy research & backtesting
│   ├── strategies/        # BaseStrategy, OpenGateStrategy
│   ├── data/              # yfinance & Polygon.io fetchers
│   ├── examples/          # Backtest scripts (yfinance, polygon)
│   ├── backtest.py        # BacktestRunner (metrics, walk-forward, Monte Carlo)
│   ├── optimize.py        # Grid search parameter optimization
│   └── report.py          # Performance report generation
├── shared/core/           # Shared bot infrastructure
│   ├── risk_manager.py    # RiskGuardrails + RiskManager
│   ├── execution.py       # ExecutionHandler + PaperExecutionHandler
│   ├── state.py           # SQLite-backed StateStore
│   ├── orchestrator.py    # BotOrchestrator + GlobalRiskMonitor
│   ├── metrics.py         # MetricsCollector
│   └── alerts.py          # AlertManager (Discord/Slack webhooks)
├── bots/                  # Per-bot instances
│   └── nq_open_gate_001/  # NQ Open Gate bot
│       ├── bot.json       # Configuration
│       ├── main.py        # Entry point
│       ├── state/         # Bot-specific SQLite state
│       └── logs/          # Log files
├── scripts/               # Operational utilities
│   ├── bot_status.py      # Health checks
│   ├── pnl_report.py      # P&L reporting
│   ├── emergency_stop.py  # Kill switch
│   ├── reset_daily_stats.py
│   └── create_bot.py      # New bot scaffolding
├── dashboard/             # Streamlit monitoring app
├── tests/                 # Unit & integration tests
└── Dockerfile.bot         # Bot containerization
```

## Strategy: Open Gate

The Open Gate strategy detects the first 5-minute candle range at market open (9:30 ET), waits for a breakout through gate_high or gate_low, confirms with a retest and candle pattern (wick rejection or engulfing), then enters with defined stop loss and take profit.

## Configuration

Copy `env.example` to `.env` and add your API keys:

```bash
cp env.example .env
```

## Adding a New Bot

```bash
uv run python scripts/create_bot.py --id es-momentum-001 --strategy momentum --ticker ES=F
```

## Deployment

```bash
# Build and start bots with Docker
docker compose up -d
```
