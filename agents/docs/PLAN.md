# Trading Bot Farm — Implementation Plan

## Overview

This plan outlines how to build a multi-bot trading system where each bot runs a focused, backtested strategy on a specific ticker or index. Each bot operates with its own wallet/account, has explicit loss limits, and executes autonomously.

---

## Phase 1: Strategy Library & Backtesting Framework

### 1.1 Set Up Unified Backtest Environment

```bash
mkdir -p ~/trading/bot-farm/{backtest,bots,shared}
cd ~/trading/bot-farm/backtest
uv init
uv add pandas numpy vectorbt backtesting yfinance ta
uv add --dev pytest pytest-cov black ruff
```

### 1.2 Implement Strategy Registry

Create a standardized interface for all strategies so backtesting and execution share the same logic:

```
backtest/
├── strategies/
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   ├── open_gate.py         # NQ/ES futures strategy
│   ├── momentum.py          # RSI/VWAP momentum
│   ├── mean_reversion.py    # Bollinger band bounce
│   └── grid.py              # Grid trading
├── data/
│   └── fetcher.py           # yfinance, polygon, IBKR
├── backtest.py              # Run backtests per strategy
├── optimize.py              # Parameter optimization
└── report.py                # Performance reports
```

**Key Principle:** *The exact same strategy class runs in backtest and live trading.* No code duplication.

### 1.3 Backtest Every Strategy Before Deployment

| Strategy | Asset Class | Data Needed | Min Backtest Period |
|----------|-------------|-------------|---------------------|
| Open Gate | Index Futures (NQ, ES) | 1-min bars, 2+ years | 500 sessions |
| RSI Momentum | Equities, Crypto | 5-min bars, 1+ year | 252 sessions |
| Mean Reversion | Equities | 1-min bars, 2+ years | 500 sessions |
| Grid | Crypto, FX | Tick/1-min, 6+ months | 180 days |
| VWAP Deviation | Large-cap equities | 1-min bars, 1+ year | 252 sessions |

**Backtest Checklist:**
- [ ] Walk-forward analysis (train/test split)
- [ ] Monte Carlo simulation on trade distribution
- [ ] Sensitivity analysis on key parameters
- [ ] Slippage and commission modeling
- [ ] Out-of-sample validation (last 6 months unseen)

### 1.4 Strategy Scoring for Bot Deployment

Only strategies meeting these thresholds graduate to bot deployment:

| Metric | Minimum Threshold | Target |
|--------|-------------------|--------|
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 15% | < 10% |
| Win Rate | > 45% | > 55% |
| Profit Factor | > 1.3 | > 1.5 |
| Expectancy | > $0 per trade | > $10 per trade |
| Calmar Ratio | > 1.0 | > 2.0 |

---

## Phase 2: Bot Architecture

### 2.1 Bot Design Principles

Each bot is **lightweight, single-purpose, and isolated**:

```
┌─────────────────────────────────────────────────────────────┐
│  Bot Instance (per strategy + ticker)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Strategy   │  │   Risk       │  │   Execution  │      │
│  │   Logic      │  │   Manager    │  │   Handler    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           │                                  │
│                    ┌──────────────┐                         │
│                    │   State      │                         │
│                    │   Store      │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Bot Configuration Template

Each bot gets a JSON config file:

```json
{
  "bot_id": "nq-open-gate-001",
  "strategy": "open_gate",
  "ticker": "NQ=F",
  "timeframe": "1m",
  "market_session": "RTH",
  "funds": {
    "account": "sub-account-001",
    "max_allocation_usd": 50000,
    "position_size_max_contracts": 2
  },
  "risk_limits": {
    "max_daily_loss_usd": 1000,
    "max_daily_loss_pct": 0.02,
    "max_drawdown_pct": 0.05,
    "max_trades_per_day": 5,
    "emergency_stop": true
  },
  "parameters": {
    "gate_candle_minutes": 5,
    "atr_multiplier": 1.5,
    "min_risk_reward": 2.0
  },
  "data_source": "polygon",
  "execution": "ibkr"
}
```

### 2.3 Bot Directory Structure

```
bots/
├── shared/                    # Shared code (read-only)
│   ├── core/
│   │   ├── strategy_base.py
│   │   ├── risk_manager.py
│   │   ├── execution.py
│   │   └── state.py
│   ├── data/
│   │   └── data_feed.py
│   └── utils/
│       └── helpers.py
├── nq-open-gate-001/          # Instance folder (per bot)
│   ├── bot.json               # Config
│   ├── main.py                # Entry point (thin wrapper)
│   ├── state/                 # Bot-specific state
│   │   ├── positions.json
│   │   ├── trades.csv
│   │   └── daily_stats.json
│   └── logs/                  # Bot-specific logs
├── es-momentum-001/
├── btc-grid-001/
└── ... (one folder per bot)
```

---

## Phase 3: Fund & Risk Management

### 3.1 Wallet/Account Structure

**Option A: Sub-accounts (Recommended for IBKR)**
- Master account with sub-accounts per bot
- Each bot sees only its allocated balance
- Master dashboard aggregates P&L
- Built-in isolation prevents cross-contamination

**Option B: Single Account with Synthetic Allocation**
- All bots share one account
- Risk manager tracks notional allocation per bot
- Prevents over-leverage across bots

**Option C: Separate API Keys (Crypto)**
- Each bot has dedicated exchange API key
- IP whitelisting per bot
- Easy revocation if needed

### 3.2 Risk Limits Per Bot

Hard-coded guards (cannot be overridden by bot):

```python
@dataclass(frozen=True)
class RiskGuardrails:
    # Daily loss limits
    max_daily_loss_usd: float      # e.g., $1000
    max_daily_loss_pct: float      # e.g., 2% of allocation
    
    # Drawdown limits
    max_drawdown_pct: float        # e.g., 5% from peak
    
    # Trade frequency
    max_trades_per_day: int        # e.g., 5
    min_time_between_trades: int   # e.g., 300 seconds
    
    # Position sizing
    max_position_size: float       # e.g., 2 contracts
    max_notional_exposure: float   # e.g., $50k
    
    # Emergency controls
    emergency_stop: bool = True    # Kill switch
    circuit_breaker_after_loss: int  # Stop after N consecutive losses
```

### 3.3 Global Risk Orchestrator

A supervisor process monitors all bots and can:
- Kill all bots if total portfolio drawdown exceeds limit
- Reduce position sizes during high volatility
- Pause trading during news events
- Rotate capital between bots based on performance

---

## Phase 4: Deployment Options

### 4.1 Deployment Matrix

| Hardware | Max Bots | Best For | Setup |
|----------|----------|----------|-------|
| Raspberry Pi 5 (16GB) | 20-50 | Lightweight rule-based bots | Docker containers |
| VPS (4GB RAM) | 10-15 | Remote execution, 24/7 | systemd + tmux |
| Workstation (32GB) | 50-100 | ML-enhanced bots, heavy compute | Kubernetes (k3s) |
| Cloud (AWS/GCP) | 100+ | Auto-scaling, global distribution | EKS/GKE |

### 4.2 Containerized Deployment

Each bot runs in its own Docker container:

```dockerfile
# Dockerfile.bot
FROM python:3.11-slim
WORKDIR /app
COPY shared/ ./shared/
COPY bots/${BOT_ID}/bot.json ./
CMD ["python", "-m", "shared.core.bot_runner", "--config", "bot.json"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  nq-open-gate-001:
    build:
      context: .
      dockerfile: Dockerfile.bot
      args:
        BOT_ID: nq-open-gate-001
    environment:
      - POLYGON_API_KEY=${POLYGON_API_KEY}
      - IBKR_ACCOUNT=${IBKR_ACCOUNT_001}
    volumes:
      - ./bots/nq-open-gate-001/state:/app/state
      - ./bots/nq-open-gate-001/logs:/app/logs
    restart: unless-stopped
  
  es-momentum-001:
    # ... similar
```

### 4.3 Orchestration with Supervisor

Simple Python orchestrator for Pi/VPS deployment:

```python
# shared/core/orchestrator.py
class BotOrchestrator:
    def __init__(self, bot_configs: list[Path]):
        self.bots: dict[str, BotProcess] = {}
        self.risk_monitor = GlobalRiskMonitor()
    
    def start_all(self):
        for config in bot_configs:
            self.spawn_bot(config)
    
    def spawn_bot(self, config: Path):
        bot_id = json.load(config.open())["bot_id"]
        process = subprocess.Popen([
            sys.executable, "-m", "shared.core.bot_runner",
            "--config", str(config)
        ])
        self.bots[bot_id] = BotProcess(
            id=bot_id,
            process=process,
            config=config
        )
    
    def monitor(self):
        while True:
            for bot in self.bots.values():
                if not bot.process.poll():
                    self.handle_bot_crash(bot)
            
            if self.risk_monitor.check_global_limits():
                self.emergency_stop_all()
            
            time.sleep(5)
```

---

## Phase 5: Execution & Monitoring

### 5.1 Execution Flow

```
Data Feed → Strategy Signal → Risk Check → Position Sizing → Order → Fill → State Update
```

Each step is asynchronous and logged:
1. **Data Feed** - WebSocket or polling, normalized OHLCV
2. **Strategy Signal** - Entry/exit logic from backtested code
3. **Risk Check** - Verify against daily limits, drawdown, exposure
4. **Position Sizing** - Calculate contracts/shares based on risk
5. **Order** - Submit to broker (IBKR, Alpaca, CCXT for crypto)
6. **Fill** - Process fill confirmation, update P&L
7. **State Update** - Persist to SQLite/Redis

### 5.2 Monitoring Dashboard

Lightweight web dashboard (Streamlit or FastAPI + React):

| View | Purpose |
|------|---------|
| Bot Status | Running/stopped, last signal, open positions |
| P&L Summary | Daily, weekly, monthly performance per bot |
| Risk Metrics | Current drawdown, daily loss used, position sizes |
| Trade Log | All fills with timestamps and P&L |
| Alerts | Risk limit breaches, crashes, anomalous behavior |

### 5.3 Alerting

- **Discord/Slack webhook** for trade notifications
- **Email/SMS** for risk limit breaches
- **PagerDuty** for critical failures (optional)

---

## Phase 6: Implementation Roadmap

### Week 1-2: Foundation
- [ ] Set up backtest environment with `uv`
- [ ] Implement base strategy class
- [ ] Port Open Gate strategy from ALGO.md
- [ ] Implement data fetcher (yfinance → Polygon)
- [ ] Run first backtests, verify metrics

### Week 3: Strategy Expansion
- [ ] Implement momentum strategy (RSI/VWAP)
- [ ] Implement mean reversion (Bollinger)
- [ ] Backtest all strategies, score them
- [ ] Document which strategies graduate to bots

### Week 4: Bot Framework
- [ ] Create shared core library
- [ ] Implement risk manager with hard limits
- [ ] Build state persistence (SQLite)
- [ ] Create execution handlers (IBKR, Alpaca)

### Week 5: First Bot
- [ ] Create `nq-open-gate-001` bot config
- [ ] Implement thin bot wrapper
- [ ] Paper trade on IBKR
- [ ] Verify execution matches backtest assumptions

### Week 6: Multi-Bot Deployment
- [ ] Create orchestrator
- [ ] Deploy 3-5 bots on Pi or VPS
- [ ] Build monitoring dashboard
- [ ] Add alerting

### Week 7+: Scaling
- [ ] Gradually add more bots
- [ ] Optimize resource usage
- [ ] Add global risk controls
- [ ] Consider auto-scaling based on performance

---

## Phase 7: Operational Runbook

### Daily Operations

```bash
# Check bot health
./scripts/bot_status.py

# View P&L summary
./scripts/pnl_report.py --today

# Review any alerts
./scripts/alerts.py --unacknowledged
```

### Adding a New Bot

1. Backtest strategy on target ticker
2. Verify metrics exceed thresholds
3. Create bot config JSON
4. Allocate funds in broker sub-account
5. `./scripts/create_bot.py --config path/to/bot.json`
6. Start in paper trading mode
7. Verify after 1 week, then go live

### Emergency Procedures

```bash
# Kill all bots immediately
./scripts/emergency_stop.py --all

# Kill specific bot
./scripts/emergency_stop.py --bot nq-open-gate-001

# Reset daily stats (start of trading day)
./scripts/reset_daily_stats.py
```

---

## Phase 8: Enhanced Bot Architecture (Detailed)

### 8.1 Bot Process Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Bot Lifecycle                                    │
├─────────────────────────────────────────────────────────────────────┤
│  1. INIT: Load config, connect to broker, initialize state         │
│  2. PRE-MARKET: Fetch data, check risk limits, prepare session     │
│  3. MARKET: Monitor data feed, generate signals, execute trades    │
│  4. POST-MARKET: Close positions, generate report, update stats    │
│  5. ERROR: Log error, attempt restart, alert on repeated failures  │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 State Machine for Each Bot

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class BotState(Enum):
    INITIALIZING = "initializing"
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MARKET_CLOSED = "market_closed"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    CRASHED = "crashed"
    SHUTDOWN = "shutdown"

@dataclass
class BotStatus:
    bot_id: str
    state: BotState
    last_signal_time: Optional[datetime]
    open_position: Optional[dict]  # {direction, entry, quantity}
    daily_pnl: float
    daily_trades: int
    cumulative_pnl: float
    last_error: Optional[str]
    uptime_seconds: int
```

### 8.3 Strategy Interface (Base Class)

```python
# backtest/strategies/base.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    name: str
    parameters: Dict[str, Any]
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        pass
    
    @abstractmethod
    def on_data(self, data: pd.DataFrame) -> Optional[dict]:
        """
        Process new data and return signal if any.
        
        Returns:
            dict with keys: signal_type (entry/exit), direction (long/short),
                           entry_price, stop_loss, take_profit, or None
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset strategy state for new session."""
        pass
    
    @abstractmethod
    def get_state(self) -> dict:
        """Return strategy state for persistence."""
        pass
    
    @abstractmethod
    def set_state(self, state: dict):
        """Restore strategy state from persistence."""
        pass
    
    def validate_signal(self, signal: dict) -> bool:
        """Validate signal before execution."""
        required_keys = {'signal_type', 'direction', 'entry_price', 'stop_loss', 'take_profit'}
        return required_keys.issubset(signal.keys())
```

### 8.4 Risk Manager Implementation

```python
# shared/core/risk_manager.py
from dataclasses import dataclass
from typing import Optional
from datetime import date

@dataclass
class RiskState:
    daily_start_equity: float
    daily_end_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_losses: int = 0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    date: date = None

class RiskManager:
    """Manages risk limits for a single bot."""
    
    def __init__(self, config: dict, initial_capital: float):
        self.config = config
        self.initial_capital = initial_capital
        self.state = RiskState(
            daily_start_equity=initial_capital,
            peak_equity=initial_capital,
            date=date.today()
        )
    
    def check_pre_trade(self, potential_loss: float) -> tuple[bool, list[str]]:
        """
        Check if trade can be executed.
        
        Returns: (allowed, list_of_reasons_if_not_allowed)
        """
        reasons = []
        
        # Daily loss limit
        if self.state.daily_pnl <= -self.config['max_daily_loss_usd']:
            reasons.append("Daily loss limit reached")
        
        # Daily loss percentage
        if self.state.daily_pnl / self.initial_capital <= -self.config['max_daily_loss_pct']:
            reasons.append("Daily loss percentage limit reached")
        
        # Max drawdown
        if self.state.max_drawdown >= self.config['max_drawdown_pct']:
            reasons.append("Maximum drawdown limit reached")
        
        # Max trades per day
        if self.state.daily_trades >= self.config['max_trades_per_day']:
            reasons.append("Max trades per day reached")
        
        # Max position size
        if potential_loss > self._calculate_max_exposure():
            reasons.append("Position size exceeds limit")
        
        return len(reasons) == 0, reasons
    
    def record_trade(self, pnl: float, is_loss: bool):
        """Record trade outcome and update risk state."""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        
        if is_loss:
            self.state.daily_losses += 1
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        # Update drawdown
        current_equity = self.state.daily_start_equity + self.state.daily_pnl
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown
    
    def should_stop_trading(self) -> Optional[str]:
        """Check if trading should stop. Returns reason if yes."""
        if self.state.daily_pnl <= -self.config['max_daily_loss_usd']:
            return f"Daily loss limit ({self.config['max_daily_loss_usd']}) reached"
        
        if self.state.consecutive_losses >= self.config['circuit_breaker_after_loss']:
            return f"Circuit breaker triggered ({self.state.consecutive_losses} losses)"
        
        if self.state.max_drawdown >= self.config['max_drawdown_pct']:
            return f"Maximum drawdown ({self.config['max_drawdown_pct']*100}%)" + " reached"
        
        return None
```

### 8.5 Execution Handler

```python
# shared/core/execution.py
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"
    filled_quantity: int = 0
    filled_price: Optional[float] = None

class ExecutionHandler(ABC):
    """Abstract base for execution handlers."""
    
    @abstractmethod
    async def connect(self):
        """Connect to broker."""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        pass
    
    @abstractmethod
    async def get_open_orders(self) -> list[Order]:
        """Get all open orders."""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> dict:
        """Get account balance and buying power."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Get current positions."""
        pass
    
    async def close_position(self, symbol: str, quantity: int) -> Order:
        """Close a position by executing opposite order."""
        # Implementation
        pass
```

### 8.6 Bot Runner

```python
# shared/core/bot_runner.py
import asyncio
import json
from pathlib import Path
from datetime import datetime, time

class BotRunner:
    """Main entry point for a bot instance."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.bot_id = self.config['bot_id']
        self.state = BotState.INITIALIZING
        
        # Initialize components
        self.strategy = self._create_strategy()
        self.risk_manager = self._create_risk_manager()
        self.execution = self._create_execution()
        self.state_store = self._create_state_store()
    
    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return json.load(f)
    
    def _create_strategy(self):
        """Create strategy instance from config."""
        strategy_type = self.config['strategy']
        params = self.config.get('parameters', {})
        
        if strategy_type == 'open_gate':
            from strategies.open_gate import OpenGateStrategy
            return OpenGateStrategy(params)
        elif strategy_type == 'momentum':
            from strategies.momentum import MomentumStrategy
            return MomentumStrategy(params)
        # ... other strategies
        else:
            raise ValueError(f"Unknown strategy: {strategy_type}")
    
    async def run(self):
        """Main bot loop."""
        await self.execution.connect()
        self.state = BotState.PRE_MARKET
        
        while True:
            now = datetime.now()
            
            # Check if market is open
            if not self._is_market_open(now):
                await asyncio.sleep(60)  # Check every minute
                continue
            
            self.state = BotState.MARKET_OPEN
            
            try:
                # Get latest data
                data = await self.execution.get_latest_prices(self.config['ticker'])
                
                # Generate signal
                signal = self.strategy.on_data(data)
                
                if signal:
                    # Check risk
                    allowed, reasons = self.risk_manager.check_pre_trade(
                        signal.get('potential_loss', 0)
                    )
                    
                    if allowed:
                        # Execute
                        order = Order(
                            order_id=self._generate_order_id(),
                            symbol=self.config['ticker'],
                            side=OrderSide.BUY if signal['direction'] == 'long' else OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=self.config['funds']['position_size_max_contracts'],
                            limit_price=signal['entry_price'],
                            stop_price=signal['stop_loss']
                        )
                        
                        filled_order = await self.execution.submit_order(order)
                        
                        if filled_order.status == 'filled':
                            self.risk_manager.record_trade(0, False)  # Record trade
                            await self.state_store.record_trade(filled_order)
                    else:
                        print(f"Risk check failed: {reasons}")
            
            except Exception as e:
                print(f"Error in bot loop: {e}")
                self.state = BotState.CRASHED
                await self._handle_error(e)
                return
            
            await asyncio.sleep(self.config.get('poll_interval', 5))
    
    def _is_market_open(self, now: datetime) -> bool:
        """Check if market is currently open."""
        market_open = time.fromisoformat(self.config['market_session']['open'])
        market_close = time.fromisoformat(self.config['market_session']['close'])
        
        # Add pre-market and post-market handling if needed
        return market_open <= now.time() <= market_close
```

---

## Phase 9: Data Pipeline Architecture

### 9.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Pipeline                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Source (yfinance/Polygon/IBKR) → Raw Data → Normalizer →          │
│  Cache (SQLite/Redis) → Strategy Feed → Strategy Logic             │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Data Fetcher Interface

```python
# backtest/data/fetcher.py
from abc import ABC, abstractmethod
from typing import Union, AsyncIterator
import pandas as pd

class DataFetcher(ABC):
    """Base class for data fetchers."""
    
    name: str
    supports_websocket: bool = False
    
    @abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1m"
    ) -> pd.DataFrame:
        """Fetch historical data."""
        pass
    
    @abstractmethod
    async def fetch_live_tick(self, symbol: str):
        """Fetch live tick data (if websocket supported)."""
        pass
    
    @abstractmethod
    async def fetch_live_bars(self, symbol: str) -> AsyncIterator[dict]:
        """Fetch live bar data (if websocket supported)."""
        pass
    
    def normalize(self, raw_data: dict, symbol: str) -> pd.DataFrame:
        """Normalize raw data to standard OHLCV format."""
        # Implementation
        pass
```

### 9.3 Data Caching Strategy

```python
# backtest/data/cache.py
import sqlite3
from datetime import datetime
from typing import Optional
import pandas as pd

class DataCache:
    """SQLite-based data cache."""
    
    def __init__(self, db_path: str = "data/cache/data.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT,
                    timestamp TIMESTAMP,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON price_data(symbol, timestamp)")
    
    def save_bars(self, symbol: str, data: pd.DataFrame):
        """Save bars to cache."""
        data['symbol'] = symbol
        data.reset_index().to_sql('price_data', self.conn, if_exists='append', index=False)
    
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: str = "1m"
    ) -> pd.DataFrame:
        """Get cached bars."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        return pd.read_sql_query(query, self.conn, params=[symbol, start, end])
```

---

## Phase 10: Performance Monitoring & Alerts

### 10.1 Metrics to Track

```python
# shared/core/metrics.py
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class BotMetrics:
    bot_id: str
    timestamp: datetime
    
    # Performance
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Activity
    total_trades: int
    today_trades: int
    open_positions: int
    
    # Risk
    current_drawdown: float
    daily_loss: float
    daily_loss_pct: float

class MetricsCollector:
    """Collects and reports bot metrics."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[BotMetrics]] = {}
    
    def record(self, bot_id: str, metrics: BotMetrics):
        """Record metrics for a bot."""
        if bot_id not in self.metrics_history:
            self.metrics_history[bot_id] = []
        self.metrics_history[bot_id].append(metrics)
    
    def get_summary(self, bot_id: str) -> dict:
        """Get summary metrics for a bot."""
        history = self.metrics_history.get(bot_id, [])
        if not history:
            return {}
        
        latest = history[-1]
        return {
            "bot_id": bot_id,
            "timestamp": latest.timestamp.isoformat(),
            "total_return_pct": latest.total_return * 100,
            "sharpe_ratio": latest.sharpe_ratio,
            "max_drawdown_pct": latest.max_drawdown * 100,
            "total_trades": latest.total_trades,
            "win_rate_pct": latest.win_rate * 100,
        }
```

### 10.2 Alert System

```python
# shared/core/alerts.py
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import smtplib
from email.message import EmailMessage

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    bot_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool = False

class AlertManager:
    """Manages alerts across all bots."""
    
    def __init__(self, config: dict):
        self.config = config
        self.alerts: list[Alert] = []
    
    def raise_alert(self, alert: Alert):
        """Raise an alert and send notification."""
        self.alerts.append(alert)
        
        if alert.severity == AlertSeverity.CRITICAL:
            self._send_notification(alert)
        elif alert.severity == AlertSeverity.WARNING:
            # Could be rate-limited or logged
            self._log_alert(alert)
    
    def _send_notification(self, alert: Alert):
        """Send notification (Discord/Slack/Email)."""
        # Implementation for webhook notifications
        pass
    
    def _log_alert(self, alert: Alert):
        """Log alert to file."""
        with open(f"logs/alerts/{alert.bot_id}.log", "a") as f:
            f.write(f"{alert.timestamp.isoformat()} [{alert.severity.value}] {alert.title}: {alert.message}\n")
```

---

## Phase 11: Deployment Considerations

### 11.1 Resource Requirements

| Component | RAM | CPU | Storage | Notes |
|-----------|-----|-----|---------|-------|
| Bot (idle) | 20MB | 1% | 10MB | Basic runtime |
| Bot (active) | 50MB | 5-10% | 10MB | Data fetching, strategy calc |
| Orchestrator | 100MB | 5% | 50MB | Manages multiple bots |
| Cache (Redis) | 500MB | 10% | 1GB | Price data cache |
| DB (SQLite) | 10MB | 1% | 10GB | Trade logs, state |
| Dashboard | 100MB | 2% | 100MB | Streamlit/FastAPI |

### 11.2 Scaling Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Scaling Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1: Single Pi (10 bots) → SQLite, local logging              │
│  Stage 2: Multiple Pis (50 bots) → Redis, shared logging           │
│  Stage 3: VPS/K8s (100+ bots) → Cloud DB, monitoring stack         │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.3 High Availability

- **Bot-level**: Auto-restart on crash (supervisor/healthcheck)
- **Data-level**: Write-ahead logging, periodic snapshots
- **Network-level**: Retry with exponential backoff
- **Broker-level**: Use IBKR's TWS failover or multiple API keys

---

## Phase 12: Testing Strategy

### 12.1 Unit Tests

```python
# tests/test_open_gate.py
import pytest
from strategies.open_gate import OpenGateStrategy

@pytest.fixture
def strategy():
    return OpenGateStrategy({"gate_minutes": 5, "stop_buffer": 1.0})

def test_gate_detection(strategy):
    # Test gate detection on sample data
    pass

def test_breakout_detection(strategy):
    # Test breakout detection
    pass

def test_retest_detection(strategy):
    # Test retest detection
    pass
```

### 12.2 Integration Tests

```python
# tests/test_bot_integration.py
import pytest
from bot_orchestrator import BotOrchestrator

@pytest.mark.asyncio
async def test_bot_lifecycle():
    orchestrator = BotOrchestrator(["config/bots/nq-open-gate.json"])
    await orchestrator.start_all()
    
    # Wait for bot to start
    await asyncio.sleep(10)
    
    # Check bot state
    status = await orchestrator.get_bot_status("nq-open-gate-001")
    assert status["state"] == "MARKET_OPEN"
    
    await orchestrator.stop_all()
```

### 12.3 Backtest Validation

```python
# tests/test_backtest_validation.py
def test_backtest_signature():
    """Verify backtest produces consistent results."""
    # Run same backtest twice, compare results
    pass

def test_walk_forward_consistency():
    """Verify walk-forward analysis works correctly."""
    # Run walk-forward, verify train/test split
    pass
```

---

## Phase 13: Production Readiness Checklist

### 13.1 Pre-Deployment

- [ ] All strategies backtested with 2+ years of data
- [ ] Walk-forward analysis completed
- [ ] Monte Carlo simulation performed
- [ ] Paper trading for 2+ weeks
- [ ] All risk limits tested
- [ ] Emergency stop tested
- [ ] Alert system configured and tested

### 13.2 Deployment

- [ ] Sub-accounts created for each bot
- [ ] API keys configured securely
- [ ] Environment variables set
- [ ] Docker containers built and tested
- [ ] Monitoring dashboard deployed
- [ ] Alert channels configured

### 13.3 Post-Deployment

- [ ] Monitor first 24 hours closely
- [ ] Verify P&L matches expectations
- [ ] Check risk limits are enforced
- [ ] Review trade logs for anomalies
- [ ] Document any issues and fixes

---

## Phase 14: Troubleshooting Guide

### 14.1 Common Issues

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Bot doesn't execute trades | Strategy not generating signals | Check strategy logs, verify data |
| Bot crashes immediately | Config error or missing dependency | Check error logs, validate config |
| Incorrect position sizing | Risk manager misconfigured | Verify allocation and limits |
| Missing data | Data source API limit exceeded | Check API status, add rate limiting |
| Large drawdown | Market condition changed | Review strategy, consider pause |

### 14.2 Debug Commands

```bash
# View bot logs
tail -f logs/bots/nq-open-gate-001/bot.log

# Check bot status
curl http://localhost:8000/api/bot/nq-open-gate-001/status

# Restart bot
docker restart nq-open-gate-001

# Reset daily stats
curl -X POST http://localhost:8000/api/bot/nq-open-gate-001/reset_daily
```

---

## Phase 15: Next Steps for Implementation

1. **Setup Project Structure**
   ```bash
   mkdir -p ~/trading/bot-farm/{backtest,bots,shared,scripts}
   cd ~/trading/bot-farm/backtest
   uv init
   uv add pandas numpy vectorbt yfinance ta
   ```

2. **Implement Base Strategy Class**
   - Create `backtest/strategies/base.py`
   - Define abstract methods for strategy interface

3. **Port Open Gate Strategy**
   - Implement in `backtest/strategies/open_gate.py`
   - Ensure compatibility with base class

4. **Build Backtesting Framework**
   - Implement `backtest/backtest.py`
   - Add metrics calculation
   - Create report generator

5. **Create Shared Core Library**
   - Implement risk manager
   - Create execution handler base
   - Build state store

6. **Build First Bot**
   - Create `bots/nq-open-gate-001/`
   - Write config JSON
   - Test in paper mode

7. **Deploy Orchestrator**
   - Create `orchestrator.py`
   - Test multi-bot management

8. **Add Monitoring**
   - Build dashboard (Streamlit)
   - Configure alerts
   - Deploy to server

9. **Graduate to Live**
   - Start with small allocation
   - Monitor closely for 1 week
   - Gradually increase autonomy

---

*This plan is a living document. Last updated: 2026-04-04*

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `backtest/strategies/base.py` | Abstract strategy interface |
| `backtest/strategies/open_gate.py` | Open Gate implementation |
| `backtest/backtest.py` | Backtest runner |
| `shared/core/risk_manager.py` | Risk limit enforcement |
| `shared/core/execution.py` | Broker order handling |
| `shared/core/state.py` | Persistence layer |
| `shared/core/orchestrator.py` | Bot supervisor |
| `bots/*/bot.json` | Per-bot configuration |
| `scripts/create_bot.py` | Bot scaffolding |
| `scripts/bot_status.py` | Health monitoring |

---

## Success Criteria

- [ ] All deployed bots have passed 2+ years backtest with Sharpe > 1.0
- [ ] Each bot has explicit daily loss limit enforced
- [ ] Bots run isolated with dedicated funds
- [ ] Global risk monitor can emergency stop all
- [ ] Paper trading validates execution for 2+ weeks before live
- [ ] Monitoring dashboard shows real-time P&L and risk metrics

