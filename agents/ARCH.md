# Open Gate Strategy — Autonomous AI Trading System Architecture

## Executive Summary

This document outlines a complete architecture for building an autonomous AI trading system that sources data, backtests strategies, and executes trades — all orchestrated by AI agents without human intervention. The focus is on index futures (NQ, ES) given the Open Gate Strategy specification, though the architecture generalizes to other assets.

---

## 1. The Autonomous Trading Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AUTONOMOUS TRADING AGENT                          │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   DATA   │───▶│   RESEARCH    │───▶│  BACKTEST   │───▶│ EXECUTE  │ │
│  │  SOURCE  │    │   & SIGNAL    │    │  & OPTIMIZE │    │ & MONITOR│ │
│  └──────────┘    └──────────────┘    └─────────────┘    └──────────┘ │
│       │                │                    │                  │       │
│       └────────────────┴────────────────────┴──────────────────┘     │
│                    SHARED STATE / MEMORY STORE                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The entire pipeline can be treated as a stateful workflow where results from each stage inform the next. An AI agent orchestrates transitions, makes decisions about parameter changes, and handles failures autonomously.

---

## 2. Data Source Options

### 2.1 Comparison Matrix

| Provider | Data Type | Intraday | Futures | Free Tier | API | Latency | Quality |
|----------|-----------|----------|---------|-----------|-----|---------|---------|
| **Polygon.io** | Stocks, Futures, Crypto | Tick → 1min | ✅ ES, NQ | ❌ 2500 credits/mo | REST + WebSocket | Low | High |
| **Alpaca** | Stocks, Crypto | 1min | ❌ | ✅ | REST + WebSocket | Low | High |
| **Interactive Brokers** | Stocks, Futures, Forex | Tick | ✅ | ❌ | Client Portal / Gateway | Very Low | Very High |
| **Tradestation** | Stocks, Futures | Tick | ✅ | ❌ | REST | Low | High |
| **yfinance** | Stocks, ETF, Indices | 1min (delayed) | ⚠️ Limited | ✅ | Python only | High | Medium |
| **Alpha Vantage** | Stocks, FX | 1min (premium) | ❌ | ✅ 25 req/day | REST | Medium | High |
| **Tiingo** | Stocks, Mutual Funds | 1min | ❌ | 500/day free | REST | Medium | High |
| **QuantConnect** | Stocks, Futures, Crypto | Tick | ✅ | ✅ with limits | LEAN Engine | Medium | Very High |
| **IEX Cloud** | Stocks, Forex | 1min+ | ❌ | $29/mo minimum | REST | Low | High |
| **Nasdaq Data Link** | Stocks, Economic | Daily mostly | ⚠️ Some | ❌ | REST | Medium | High |
| **CME Group** | Futures Only | Daily + some tick | ✅ | ❌ | FTP/API | Low | Very High |
| **OANDA** | Forex, Indices | Tick | ⚠️ Limited | ✅ | REST + Streaming | Low | High |
| **FXCM** | Forex, Indices | Tick | ⚠️ Limited | ✅ | REST | Low | High |

### 2.2 Detailed Provider Analysis

#### **Polygon.io** — Recommended for AI Agents

**Why:** Clean API, excellent documentation, real-time WebSocket support, Python-first SDK (`polygon`)

**Strengths:**
- Tick-level data for stocks and futures
- REST + WebSocket for live data
- Python client is well-maintained
- Historical data accessible via REST
- Supports futures chains, tickers, trades, quotes, bars

**Weaknesses:**
- Not free (~$200/mo for productive use)
- Rate limits on free tier (2500 credits/day)

**Code Example:**
```python
from polygon import RESTClient

client = RESTClient("API_KEY")

# Fetch minute bars for NQ futures
aggs = client.get_aggs(
    "NQ:FUT",  # NQ futures contract
    1,  # multiplier
    "minute",  # timespan
    "2024-01-01",
    "2024-01-31",
    adjusted=True,
)

for agg in aggs:
    print(f"Date: {agg.timestamp}, Open: {agg.open}, High: {agg.high}, Low: {agg.low}, Close: {agg.close}")
```

---

#### **Interactive Brokers (IBKR)** — Best for Execution

**Why:** Direct market access, best-in-class execution, unified data + execution

**Strengths:**
- Unified market data and order execution
- Real-time tick data for futures
- Supports paper trading
- Competitive commission (~$0.85/contract for NQ)
- Python API via `ib_insync` or official API
- Historical data via `reqHistoricalData`

**Weaknesses:**
- Complex API (steep learning curve)
- Client Portal Gateway requires maintenance
- Data tied to trading account

**Code Example:**
```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # paper trading

# Fetch 1-minute bars for NQ
contract = Future('NQ', '202403', 'CME')

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True
)

df = util.df(bars)
print(df.head())
```

**For Autonomous Agents:** IB's API is powerful but requires careful session management. Use `ib_insync` for async operations and wrap in retry logic.

---

#### **QuantConnect** — Best for Research-to-Production

**Why:** Full research environment, LEAN engine handles both backtesting and live execution, free data (with limits)

**Strengths:**
- Jupyter-powered research environment
- Built-in data from many providers
- Unified backtest + live trading
- Supports Python, C#, F#
- Large community, extensive examples
- Paper trading and live brokerage connections

**Weaknesses:**
- Cloud execution requires QuantConnect infrastructure
- Local execution requires Docker + Lean engine setup
- Proprietary platform lock-in

**Architecture for AI Agent:**
```python
# QuantConnect research (in Jupyter)
qb = QuantBook()
nq = qb.AddFuture(Futures.MicroEminiNasdaq100)
qb.AddEquity("SPY")  # for context

# Get historical data
history = qb.History(nq.Symbol, timedelta(days=30), Resolution.Minute)

# Build strategy
# ... custom logic ...

# Deploy to paper/live
qb.QueueCommand(...)
```

---

#### **Alpaca** — Best Free Option for Stocks/Crypto

**Why:** Commission-free, real-time WebSocket, easy to use API

**Strengths:**
- Free market data API (real-time for US equities)
- Crypto data included
- Paper trading included
- Clean REST API
- Python SDK (`alpaca-trade-api`)

**Weaknesses:**
- **No futures support** (major limitation for Open Gate)
- Limited historical data on free tier
- US equities only (plus crypto)

**Verdict:** Good for stock strategies, but not suitable for this futures strategy.

---

#### **yfinance** — Best Free Option (with caveats)

**Why:** Completely free, easy install, decent for daily/intraday

**Strengths:**
- 100% free
- No API keys needed
- Good enough for research
- Handles dividends, splits correctly

**Weaknesses:**
- **Data is delayed 15-60 minutes** for intraday
- Not reliable for real-time
- No guarantee of data quality
- Yahoo can change API without notice

**Code Example:**
```python
import yfinance as yf

# Fetch 1-minute data for NQ futures (NQ=F)
data = yf.download(
    ticker="NQ=F",
    start="2024-01-01",
    end="2024-01-31",
    interval="1m",
    prepost=True  # include pre/post market
)
```

**For AI Agents:** Use yfinance for research and backtesting, but switch to a real-time source for execution.

---

### 2.3 Futures-Specific Data Options

Since the Open Gate strategy targets NQ (Nasdaq) and ES (S&P 500) futures, you need futures-specific data:

| Source | Futures Coverage | Quality | Cost |
|--------|-----------------|---------|------|
| **CME Group** | All CME futures | Very High | Market data fees |
| **Eurekahedge** | Hedge fund indices | High | Subscription |
| **Quandl** (Nasdaq Data Link) | Some futures | High | Subscription |
| **Barchart** | Futures, commodities | High | Subscription |
| **RJO'Brien** | Futures, institutional | High | Institutional pricing |
| **Mirus Futures** | Futures, institutional | High | Institutional pricing |
| **Interactive Brokers** | All major futures | Very High | Commission-based |

**CME Group Historical Data:**
```python
# Via Polygon (better API)
aggs = client.get_aggs("NQ:FUT", 1, "minute", "2024-01-01", "2024-01-31")

# Via IBKR
bars = ib.reqHistoricalData(Future('NQ', '202403', 'CME'), ...)
```

---

## 3. AI Agent Architecture

### 3.1 Agent Design Patterns

For an autonomous trading system, we need multiple specialized agents working together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT (Primary)                        │
│  • Coordinates sub-agents                                               │
│  • Makes go/no-go trading decisions                                     │
│  • Handles risk management overrides                                    │
│  • Maintains session state                                              │
└─────────────────────────────────────────────────────────────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  DATA AGENT     │    │  RESEARCH AGENT     │    │  EXECUTION AGENT    │
│                 │    │                     │    │                     │
│ • Fetch data    │    │ • Run backtests     │    │ • Place orders      │
│ • Clean &       │    │ • Optimize params   │    │ • Monitor positions │
│   normalize     │    │ • Signal generation │    │ • Handle fills      │
│ • Update price  │    │ • Risk assessment   │    │ • Track P&L         │
│   state         │    │                     │    │                     │
└─────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### 3.2 Tool Framework Options

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **LangGraph** | Stateful, built-in persistence, designed for agents | Newer, smaller community | Production agent systems |
| **CrewAI** | Role-based agents, easy to configure | Less granular control | Structured workflows |
| **AutoGen** | Microsoft-backed, multi-agent | Complex setup | Research/experimentation |
| **LlamaIndex** | Great for RAG + data | Less agent-focused | Data-intensive agents |
| **Raw Python + Threads** | Full control, minimal deps | More code to write | Simple automations |

**Recommended: LangGraph** for production-grade autonomy with built-in checkpointing and human-in-the-loop capabilities.

---

### 3.3 Shared State Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REDIS / SQLite                               │
│                                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐  │
│  │ price_state │  │ signal_state │  │ pos_state │  │ perf_state │  │
│  │             │  │              │  │           │  │            │  │
│  │ last_price  │  │ gate_high    │  │ position  │  │ equity     │  │
│  │ last_time   │  │ gate_low     │  │ entry_pri │  │ trades     │  │
│  │ session_    │  │ broke_above  │  │ stop_loss │  │ daily_pnl  │  │
│  │   open_time │  │ waiting_     │  │ take_prof │  │ drawdown   │  │
│  │             │  │   retest     │  │           │  │            │  │
│  └─────────────┘  └──────────────┘  └───────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         │                │                   │                │
         └────────────────┴───────────────────┴────────────────┘
                              PERSISTENCE
```

**State Schema:**
```python
@dataclass
class TradingState:
    # Market state
    session_date: date
    session_open_time: datetime
    gate_high: Optional[float] = None
    gate_low: Optional[float] = None
    gate_set: bool = False
    
    # Direction state
    broke_above: bool = False
    broke_below: bool = False
    waiting_for_retest: bool = False
    retest_direction: Optional[str] = None
    
    # Position state
    position_open: bool = False
    position_direction: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Performance state
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
```

---

## 4. Full Autonomous System Implementation

### 4.1 Project Structure

```
open-gate-autonomous/
├── pyproject.toml
├── uv.lock
├── ARCH.md
├── .env                           # API keys, secrets
├── config/
│   ├── strategy.yaml              # Strategy parameters
│   ├── agents.yaml                # Agent configurations
│   └── risk.yaml                  # Risk limits
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Main coordinator
│   │   ├── data_agent.py          # Data fetching
│   │   ├── research_agent.py      # Backtesting & signals
│   │   └── execution_agent.py     # Order execution
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── open_gate.py           # Open Gate strategy
│   │   ├── indicators.py          # Technical indicators
│   │   └── confirmation.py        # Entry confirmations
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py             # Multi-source data fetching
│   │   ├── polygon_client.py      # Polygon.io adapter
│   │   ├── ibkr_client.py         # IBKR adapter
│   │   └── normalizer.py          # Data normalization
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── broker.py              # Broker abstraction
│   │   ├── ibkr_broker.py         # IBKR execution
│   │   ├── alpaca_broker.py        # Alpaca execution
│   │   └── paper_broker.py        # Paper trading
│   ├── state/
│   │   ├── __init__.py
│   │   ├── store.py               # State persistence
│   │   └── schema.py              # State definitions
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── retry.py
├── notebooks/
│   ├── research.ipynb             # Interactive research
│   └── monitoring.ipynb           # Live monitoring
├── tests/
│   ├── test_strategy.py
│   ├── test_data_pipeline.py
│   └── test_integration.py
└── logs/
    └── trades/                    # Trade logs
```

### 4.2 Core Implementation

#### Data Agent
```python
# src/agents/data_agent.py
import asyncio
from datetime import datetime, date
from typing import Optional
import pandas as pd
from polygon import RESTClient

class DataAgent:
    """Fetches and normalizes market data from multiple sources."""
    
    def __init__(self, config: dict):
        self.config = config
        self.primary_client = None
        self.fallback_clients = []
        self._setup_clients()
    
    def _setup_clients(self):
        # Primary: Polygon.io
        if polygon_key := self.config.get("polygon_api_key"):
            self.primary_client = PolygonClient(polygon_key)
        
        # Fallback: IBKR
        if self.config.get("ibkr_enabled"):
            self.fallback_clients.append(IBKRClient())
        
        # Fallback: yfinance
        self.fallback_clients.append(YFinanceClient())
    
    async def fetch_minute_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        provider: str = "polygon"
    ) -> pd.DataFrame:
        """Fetch minute bars with automatic fallback."""
        
        for client in self._get_client_order(provider):
            try:
                data = await client.fetch_bars(symbol, start, end)
                return self._normalize_data(data, symbol)
            except Exception as e:
                await self._handle_error(client, e)
                continue
        
        raise RuntimeError(f"All data sources failed for {symbol}")
    
    async def subscribe_live(
        self,
        symbol: str,
        callback: Callable[[dict], None]
    ) -> asyncio.Task:
        """Subscribe to real-time data via WebSocket."""
        ws = PolygonWebSocket(self.config["polygon_api_key"])
        await ws.subscribe(f"AM.{symbol}")  # Aggregate minute bars
        return asyncio.create_task(self._ws_listener(ws, callback))
```

#### Research Agent
```python
# src/agents/research_agent.py
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    expectancy: float
    optimal_params: dict

class ResearchAgent:
    """Handles backtesting, optimization, and signal generation."""
    
    def __init__(self, state_store: StateStore):
        self.state = state_store
        self.strategy = OpenGateStrategy()
    
    async def run_backtest(
        self,
        data: pd.DataFrame,
        params: dict
    ) -> BacktestResult:
        """Run backtest with given parameters."""
        
        self.strategy.configure(**params)
        trades = []
        equity_curve = [100_000]
        
        for idx, row in data.iterrows():
            # Update strategy state
            self.strategy.update(row)
            
            # Check for signals
            signal = self.strategy.evaluate()
            
            if signal:
                trades.append(self._execute_signal(signal, row, equity_curve[-1]))
                self.strategy.reset_signal()
            
            # Track equity
            equity_curve.append(self._calculate_equity(trades, row))
        
        return self._compute_metrics(trades, equity_curve)
    
    async def optimize(
        self,
        data: pd.DataFrame,
        param_grid: dict
    ) -> BacktestResult:
        """Grid search over parameter space."""
        
        best_result = None
        best_score = float('-inf')
        
        for params in self._generate_param_combinations(param_grid):
            result = await self.run_backtest(data, params)
            
            if result.sharpe_ratio > best_score:
                best_score = result.sharpe_ratio
                best_result = result
        
        return best_result
    
    async def generate_signal(self, latest_data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal from latest data."""
        
        self.strategy.update(latest_data.iloc[-1])
        
        # Get current state
        state = self.state.get_trading_state()
        
        if not state.gate_set and self._is_market_open():
            # Check if gate should be set (first 5 min candle complete)
            gate_data = latest_data.tail(5)
            if len(gate_data) >= 5:
                self.strategy.set_gate(
                    gate_data['high'].max(),
                    gate_data['low'].min()
                )
                self.state.update_gate(state.session_date)
        
        return self.strategy.evaluate()
```

#### Execution Agent
```python
# src/agents/execution_agent.py
import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    order_id: str
    direction: str  # "long" or "short"
    quantity: int
    order_type: str  # "market", "limit", "stop"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    created_at: datetime = None

class ExecutionAgent:
    """Handles order execution and position management."""
    
    def __init__(
        self,
        broker: BrokerAdapter,
        state_store: StateStore,
        risk_config: dict
    ):
        self.broker = broker
        self.state = state_store
        self.risk = risk_config
        self.max_position_size = risk_config["max_contracts"]
        self.max_daily_loss = risk_config["max_daily_loss_pct"] / 100
    
    async def execute_signal(self, signal: Signal) -> Order:
        """Execute a trading signal with risk checks."""
        
        # Pre-trade risk checks
        await self._pre_trade_checks(signal)
        
        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            direction=signal.direction,
            quantity=self._calculate_position_size(signal),
            order_type="limit",
            limit_price=signal.entry_price,
            stop_price=signal.stop_loss,
            created_at=datetime.now()
        )
        
        # Submit to broker
        result = await self.broker.submit_order(order)
        
        # Update state
        await self.state.update_position(
            direction=signal.direction,
            entry_price=result.filled_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        return result
    
    async def monitor_position(self, order: Order) -> asyncio.Task:
        """Monitor position and manage exits."""
        
        while order.status != OrderStatus.FILLED:
            await asyncio.sleep(1)
            order = await self.broker.get_order(order.order_id)
        
        # Start stop-loss monitoring
        task = asyncio.create_task(self._monitor_stops(order))
        return task
    
    async def _monitor_stops(self, order: Order):
        """Monitor and execute stop-loss orders."""
        
        while True:
            current_price = await self.broker.get_latest_price(order.symbol)
            state = await self.state.get_trading_state()
            
            if order.direction == "long":
                if current_price <= order.stop_price:
                    await self.broker.close_position(order)
                    await self.state.flat_position()
                    break
                
                if state.take_profit and current_price >= state.take_profit:
                    await self.broker.close_position(order)
                    await self.state.flat_position()
                    break
            
            await asyncio.sleep(5)
```

#### Orchestrator Agent
```python
# src/agents/orchestrator.py
import asyncio
from datetime import datetime, time
from typing import Optional
from dataclasses import dataclass

@dataclass
class AgentConfig:
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    pre_market_minutes: int = 15
    post_market_minutes: int = 15

class OrchestratorAgent:
    """Coordinates all sub-agents and manages the trading session."""
    
    def __init__(
        self,
        config: AgentConfig,
        data_agent: DataAgent,
        research_agent: ResearchAgent,
        execution_agent: ExecutionAgent,
        state_store: StateStore
    ):
        self.config = config
        self.data = data_agent
        self.research = research_agent
        self.execution = execution_agent
        self.state = state_store
        self.running = False
    
    async def start_session(self):
        """Begin the trading session."""
        
        self.running = True
        session_date = datetime.now().date()
        
        await self.state.start_session(session_date)
        
        # Pre-market: Fetch data, run backtests, prepare
        await self._pre_market_routine()
        
        # Market hours: Real-time monitoring and execution
        if self._is_market_hours():
            await self._live_trading_loop()
        
        # Post-market: Performance analysis, session close
        await self._post_market_routine()
    
    async def _pre_market_routine(self):
        """Prepare for trading day."""
        
        # 1. Fetch historical data
        data = await self.data.fetch_historical(
            symbol="NQ",
            days=30,
            interval="1m"
        )
        
        # 2. Run optimization with fresh data
        optimal = await self.research.optimize(
            data=data,
            param_grid={
                "stop_buffer": [0.5, 1.0, 1.5, 2.0],
                "gate_minutes": [3, 5, 7],
                "confirmation_type": ["wick", "engulfing", "structure"]
            }
        )
        
        # 3. Update strategy with optimal params
        await self.state.update_strategy_params(optimal.optimal_params)
        
        # 4. Check daily risk limits
        await self._check_risk_limits()
    
    async def _live_trading_loop(self):
        """Main real-time trading loop."""
        
        while self.running and self._is_market_hours():
            try:
                # 1. Get latest data
                latest = await self.data.get_latest_bars("NQ", count=100)
                
                # 2. Generate signal
                signal = await self.research.generate_signal(latest)
                
                if signal:
                    # 3. Execute if no position
                    state = await self.state.get_trading_state()
                    if not state.position_open:
                        order = await self.execution.execute_signal(signal)
                        await self.execution.monitor_position(order)
                
                # 4. Update state with latest prices
                await self.state.update_prices(latest)
                
                # 5. Check if session should end
                if self._should_end_session():
                    break
                
                await asyncio.sleep(5)  # 5-second polling
                
            except Exception as e:
                await self._handle_error(e)
                await asyncio.sleep(10)
    
    async def _post_market_routine(self):
        """Close out session and analyze performance."""
        
        # 1. Close any open positions at market close
        state = await self.state.get_trading_state()
        if state.position_open:
            await self.execution.close_all_positions()
        
        # 2. Generate session report
        report = await self._generate_session_report()
        
        # 3. Log everything
        await self.state.end_session(report)
        
        self.running = False
```

### 4.3 State Store Implementation
```python
# src/state/store.py
import json
import asyncio
from pathlib import Path
from datetime import date, datetime
from typing import Optional, Any
from dataclasses import asdict, dataclass

class StateStore:
    """Persistent state management for the trading system."""
    
    def __init__(self, storage_path: Path = Path("./data/state")):
        self.path = storage_path
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._memory: dict = {}
        self._load()
    
    def _load(self):
        """Load state from disk on startup."""
        state_file = self.path / "trading_state.json"
        if state_file.exists():
            with open(state_file) as f:
                self._memory = json.load(f)
    
    async def save(self):
        """Persist state to disk."""
        async with self._lock:
            with open(self.path / "trading_state.json", "w") as f:
                json.dump(self._memory, f, default=str)
    
    async def update_gate(self, session_date: date, high: float, low: float):
        """Update gate levels."""
        self._memory["gate"] = {
            "date": str(session_date),
            "high": high,
            "low": low,
            "set": True
        }
        await self.save()
    
    async def update_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ):
        """Update open position."""
        self._memory["position"] = {
            "open": True,
            "direction": direction,
            "entry": entry_price,
            "stop": stop_loss,
            "take_profit": take_profit,
            "opened_at": datetime.now().isoformat()
        }
        await self.save()
    
    async def flat_position(self):
        """Clear position."""
        self._memory["position"] = {"open": False}
        await self.save()
    
    async def get_trading_state(self) -> TradingState:
        """Get current trading state."""
        return TradingState(**self._memory.get("trading_state", {}))
    
    async def update_trading_state(self, **kwargs):
        """Update trading state fields."""
        self._memory.setdefault("trading_state", {}).update(kwargs)
        await self.save()
```

---

## 5. Data Source Integration Matrix

| Source | Data Agent | Research Agent | Execution | Best For |
|--------|------------|-----------------|-----------|----------|
| **Polygon + IBKR** | Polygon | Both | IBKR | Production |
| **Polygon + Alpaca** | Polygon | Polygon | Alpaca | Stocks/Crypto only |
| **yfinance + IBKR** | yfinance | Both | IBKR | Budget research |
| **QuantConnect** | QC | QC | QC | All-in-one platform |
| **IBKR only** | IBKR | IBKR | IBKR | Simplicity |

---

## 6. Recommended Architecture for Open Gate

Given the requirements (futures, AI agent autonomy, backtesting + execution):

### Option A: Polygon + IBKR (Production)

```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (LangGraph)                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌───────────┐   ┌────────────┐
    │ DATA    │    │ RESEARCH  │   │ EXECUTION  │
    │ Polygon │    │ vectorbt  │   │ IBKR       │
    └─────────┘    └───────────┘   └────────────┘
```

**Pros:** Best data quality + best execution  
**Cons:** Two paid services

### Option B: QuantConnect (Simplicity)

```
┌─────────────────────────────────────────────────────────────────┐
│  QuantConnect Cloud (Research + Backtest + Execution)            │
│                                                                  │
│  Jupyter Research → Lean Engine Backtest → Paper/Live Broker    │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:** Unified platform, handles everything  
**Cons:** Less flexibility, dependency on QC infrastructure

### Option C: IBKR + yfinance (Budget)

```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (LangGraph)                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌───────────┐   ┌────────────┐
    │ yfinance│    │ vectorbt  │   │ IBKR       │
    └─────────┘    └───────────┘   └────────────┘
```

**Pros:** Free data, professional execution  
**Cons:** yfinance data delays, potential reliability issues

---

## 7. Implementation Roadmap

### Phase 1: Research & Backtesting (Week 1-2)
1. Set up project with `uv`
2. Implement `OpenGateStrategy` class
3. Connect data source (Polygon or yfinance)
4. Build backtesting framework (vectorbt)
5. Run initial backtests and parameter optimization

### Phase 2: Paper Trading (Week 3-4)
1. Connect execution broker (IBKR or Alpaca)
2. Build execution agent
3. Run paper trading with live data
4. Validate execution matches backtest

### Phase 3: Autonomous Operation (Week 5-6)
1. Implement orchestrator agent
2. Build state management
3. Integrate all components
4. Add risk management and monitoring
5. Run in paper mode with full autonomy

### Phase 4: Production (Week 7+)
1. Switch to live execution
2. Start with small size
3. Monitor closely
4. Gradually increase autonomy
5. Add more sophisticated risk controls

---

## 8. Risk Management Rules (Non-Negotiable)

```python
# These should be enforced at the orchestrator level, not strategy level

MAX_DAILY_LOSS = 0.02          # 2% of equity per day
MAX_POSITION_SIZE = 5         # max NQ contracts
MAX_TRADES_PER_DAY = 10       # prevent overtrading
MIN_RISK_REWARD = 1.5         # minimum RR ratio
NO_TRADE_NEWS = True          # skip during major news events
MAX_DRAWDOWN_STOP = 0.05      # 5% equity stop trading
```

---

## 9. Key Files for Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project dependencies via uv |
| `.env` | API keys (polygon, ibkr, etc.) |
| `config/strategy.yaml` | Tunable strategy parameters |
| `config/risk.yaml` | Risk limits |
| `src/agents/orchestrator.py` | Main AI agent coordinator |
| `src/strategy/open_gate.py` | Strategy logic |
| `src/data/fetcher.py` | Multi-source data fetching |
| `src/state/store.py` | State persistence |

---

## 10. Alternative: No-Code / Low-Code Platforms

If building from scratch is too complex:

| Platform | AI Agent Capability | Futures | Backtesting | Execution |
|----------|---------------------|---------|-------------|-----------|
| **QuantConnect** | Partial (scheduled) | ✅ | ✅ Built-in | ✅ |
| **TradingView** | Via PineScript | ✅ | ✅ | Via webhooks |
| **Tradair** | Algorithmic | ✅ | ✅ | ✅ |
| **ProRealTime** | Automated | ✅ | ✅ | ✅ |
| **MetaTrader 5** | Via MQL5 | ✅ | ✅ | ✅ |
| **Jesse** | Python | ⚠️ Crypto only | ✅ | ⚠️ Crypto only |

**TradingView Alert → Webhook → Broker** is a common pattern:
```
TradingView Alert 
    → Webhook URL (your server) 
    → Python Flask/FastAPI endpoint 
    → Broker API (IBKR/Alpaca)
```

---

## 11. Summary Recommendations

| Priority | Component | Recommended Choice | Cost |
|----------|-----------|-------------------|------|
| **Data** | Minute bars | **Polygon.io** | ~$50-200/mo |
| **Research** | Backtesting | **vectorbt** + Python | Free |
| **Execution** | Futures trading | **Interactive Brokers** | ~$0.85/contract |
| **Orchestration** | AI agent framework | **LangGraph** | Free |
| **State** | Persistence | **SQLite/Redis** | Free |
| **Monitoring** | Dashboards | **Plotly Dash / Streamlit** | Free |

**Minimum Viable Autonomous System:**
1. yfinance (data) → vectorbt (backtest) → IBKR (execution) → Python scripts (orchestration)

**Production Autonomous System:**
1. Polygon (data) → vectorbt/Lean (backtest) → IBKR (execution) → LangGraph (orchestration)
