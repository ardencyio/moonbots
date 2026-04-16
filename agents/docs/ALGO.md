# Open Gate Trading Strategy — Build & Backtest Guide

## Overview

This document outlines a complete approach to building, backtesting, and validating the Open Gate Trading Strategy using Python with `uv` as the package manager.

---

## 1. Project Setup

### Initialize Project
```bash
mkdir open-gate-strategy && cd open-gate-strategy
uv init
```

### Core Dependencies
```bash
uv add pandas numpy matplotlib plotly
uv add backtesting   # Backtesting library
uv add requests pandas-datareader yfinance  # Data fetching
uv add jupyter
uv add --dev pytest pytest-cov black ruff mypy
```

### Data Dependencies
```bash
uv add ta   # Technical analysis indicators
uv add vectorbt  # Quant analysis & backtesting (optional, more powerful)
```

---

## 2. Data Acquisition

### Options

| Source | Method | Notes |
|--------|--------|-------|
| **yfinance** | `yfinance.download("NQ=F", start, end)` | Free, 1m data available |
| **Alpha Vantage** | API key required | Reliable, requires key |
| **Polygon.io** | API key required | High quality, real-time |
| **IBKR** | `ib_insync` library | Direct from broker |
| **CSV** | Manual import | If you have historical data |

### Recommended: yfinance + Vectorbt
```python
import yfinance as yf

def fetch_data(ticker: str, start: str, end: str, interval: str = "1m"):
    """Fetch minute-level data for futures."""
    data = yf.download(ticker, start=start, end=end, interval=interval, prepost=True)
    return data
```

### Data Storage
```
data/
├── raw/
│   └── NQ1!_1m_2023_2024.csv
├── processed/
│   └── ohlcv_1m.parquet
└── signals/
    └── signals.parquet
```

---

## 3. Strategy Implementation

### Core Class Structure

```python
# strategy.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class OpenGateConfig:
    session_open: str = "09:30:00"
    session_close: str = "16:00:00"
    timezone: str = "America/New_York"
    gate_candle_minutes: int = 5
    stop_buffer_ticks: float = 1.0
    atr_multiplier: float = 1.5
    min_risk_reward: float = 2.0

class OpenGateStrategy:
    def __init__(self, config: OpenGateConfig):
        self.config = config
        self.gate_high: Optional[float] = None
        self.gate_low: Optional[float] = None
        self.gate_set: bool = False
        self.broke_above: bool = False
        self.broke_below: bool = False
        self.waiting_for_retest: bool = False
        self.retest_direction: Optional[str] = None

    def reset(self):
        """Reset state for new session."""
        self.gate_high = None
        self.gate_low = None
        self.gate_set = False
        self.broke_above = False
        self.broke_below = False
        self.waiting_for_retest = False
        self.retest_direction = None

    def is_market_open(self, dt: pd.Timestamp) -> bool:
        """Check if within regular trading hours."""
        if dt.time() >= pd.Timestamp(self.config.session_open, tz=self.config.timezone).time():
            if dt.time() <= pd.Timestamp(self.config.session_close, tz=self.config.timezone).time():
                return True
        return False

    def detect_gate(self, df: pd.DataFrame, session_date: pd.Timestamp) -> pd.DataFrame:
        """Find the first 5-min candle high/low for a session."""
        mask = (
            (df.index.date == session_date.date()) &
            (df.index.time >= pd.Timestamp(self.config.session_open, tz=self.config.timezone).time()) &
            (df.index.time < pd.Timestamp(self.config.session_open, tz=self.config.timezone).time() + pd.Timedelta(minutes=5))
        )
        first_5min = df.loc[mask]
        if len(first_5min) > 0:
            self.gate_high = first_5min['High'].max()
            self.gate_low = first_5min['Low'].min()
            self.gate_set = True
        return first_5min

    def detect_breakout(self, row: pd.Series) -> None:
        """Detect if price broke through the gate."""
        if not self.gate_set:
            return

        if row['Close'] > self.gate_high and not self.broke_above:
            self.broke_above = True
            self.waiting_for_retest = True
            self.retest_direction = 'bullish'

        elif row['Close'] < self.gate_low and not self.broke_below:
            self.broke_below = True
            self.waiting_for_retest = True
            self.retest_direction = 'bearish'

    def detect_retest(self, row: pd.Series) -> bool:
        """Detect if price returned to test the broken level."""
        if not self.waiting_for_retest:
            return False

        if self.retest_direction == 'bullish':
            if row['Low'] <= self.gate_high:
                return True
        elif self.retest_direction == 'bearish':
            if row['High'] >= self.gate_low:
                return True
        return False

    def check_confirmation(self, df: pd.DataFrame, idx: int) -> dict:
        """Check for entry confirmation signals."""
        if idx < 3:
            return {"confirmed": False, "type": None}

        current = df.iloc[idx]
        prev1 = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        confirmation = {"confirmed": False, "type": None}

        # Bullish confirmation
        if self.retest_direction == 'bullish':
            # Lower wick rejection
            if (current['Low'] < prev1['Low'] and 
                (current['Close'] > current['Open'])):
                confirmation = {"confirmed": True, "type": "wick_rejection"}

            # Bullish engulfing
            if (prev1['Close'] < prev1['Open'] and 
                current['Close'] > prev1['Open'] and
                current['Open'] < prev1['Close']):
                confirmation = {"confirmed": True, "type": "bullish_engulfing"}

        # Bearish confirmation
        elif self.retest_direction == 'bearish':
            # Upper wick rejection
            if (current['High'] > prev1['High'] and 
                (current['Close'] < current['Open'])):
                confirmation = {"confirmed": True, "type": "wick_rejection"}

            # Bearish engulfing
            if (prev1['Close'] > prev1['Open'] and 
                current['Close'] < prev1['Open'] and
                current['Open'] > prev1['Close']):
                confirmation = {"confirmed": True, "type": "bearish_engulfing"}

        return confirmation

    def calculate_stop_loss(self, direction: str) -> float:
        """Calculate stop loss based on gate levels."""
        if direction == 'long':
            return self.gate_low - self.config.stop_buffer_ticks
        return self.gate_high + self.config.stop_buffer_ticks

    def calculate_take_profit(self, entry: float, stop: float, rr: float = 2.0) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(entry - stop)
        if risk == 0:
            return entry
        return entry + (risk * rr) if direction == 'long' else entry - (risk * rr)
```

---

## 4. Backtesting Framework

### Option A: Vectorbt (Recommended)

```python
# backtest_vectorbt.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from strategy import OpenGateStrategy, OpenGateConfig

def run_backtest_vectorbt(data: pd.DataFrame, config: OpenGateConfig):
    """Run backtest using vectorbt."""
    
    strategy = OpenGateStrategy(config)
    entries = []
    exits = []
    
    # Convert to 5-minute for gate detection
    data_5m = data.resample('5min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Detect gates per session
    signals = []
    for date, group in data_5m.groupby(data_5m.index.date):
        session_data = group.sort_index()
        gate = strategy.detect_gate(session_data, session_data.index[0])
        
        # For each 1m candle after gate set
        for idx, row in data.loc[data.index.date == date].iterrows():
            strategy.detect_breakout(row)
            
            if strategy.detect_retest(row):
                confirm = strategy.check_confirmation(data.loc[data.index <= idx], 
                                                      len(data.loc[data.index <= idx]) - 1)
                if confirm['confirmed']:
                    direction = strategy.retest_direction
                    stop = strategy.calculate_stop_loss(direction)
                    entry = row['Close']
                    
                    signals.append({
                        'timestamp': idx,
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'signal_type': 'entry'
                    })
                    
                    # Reset after entry
                    strategy.waiting_for_retest = False
    
    return pd.DataFrame(signals)

# Run
data = pd.read_parquet('data/processed/ohlcv_1m.parquet')
config = OpenGateConfig()
results = run_backtest_vectorbt(data, config)

# Calculate performance
portfolio = vbt.Portfolio.from_signals(
    data['Close'],
    entries=results['direction'] == 'bullish',
    exits=results['direction'] == 'bearish',
    stop=config.stop_buffer_ticks,
    freq='1min'
)

print(portfolio.stats())
portfolio.plot().show()
```

### Option B: Backtesting.py

```python
# backtest_backtesting.py
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from strategy import OpenGateStrategy, OpenGateConfig

class OpenGateBacktest(Strategy):
    config = OpenGateConfig()
    strategy = OpenGateStrategy(config)

    def init(self):
        self.data = self.data.df

    def next(self):
        dt = self.data.index[-1]
        
        # Reset on new session
        if dt.time() == pd.Timestamp(self.config.session_open, 
                                      tz=self.config.timezone).time():
            self.strategy.reset()

        # Check for entries
        if not self.strategy.waiting_for_retest:
            self.strategy.detect_breakout(self.data.iloc[-1])
        
        if self.strategy.detect_retest(self.data.iloc[-1]):
            confirm = self.strategy.check_confirmation(
                self.data, len(self.data) - 1
            )
            if confirm['confirmed']:
                if self.strategy.retest_direction == 'bullish':
                    self.buy()
                else:
                    self.sell()

# Run
bt = Backtest(data, OpenGateBacktest, cash=100_000, commission=.002)
stats, heatmap = bt.optimize(
    stop_buffer_ticks=[0.5, 1.0, 1.5, 2.0],
    gate_candle_minutes=[3, 5, 7],
    maximize='Equity Final [$]',
    return_heatmap=True
)

print(stats)
print(heatmap)
```

---

## 5. Performance Metrics

### Core Metrics to Track

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall percentage return |
| **Sharpe Ratio** | Risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / gross loss |
| **Avg Trade Duration** | Average time in trades |
| **Expectancy** | Average return per trade |
| **Calmar Ratio** | Return / max drawdown |
| **Trade Count** | Total number of trades |
| **% Profitable** | Win rate percentage |

### Equity Curve Analysis
```python
def analyze_equity_curve(equity: pd.Series) -> dict:
    """Analyze equity curve performance."""
    returns = equity.pct_change().dropna()
    
    return {
        'total_return': (equity[-1] / equity[0] - 1) * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 390),
        'max_drawdown': ((equity / equity.cummax()) - 1).min() * 100,
        'calmar_ratio': (equity[-1] / equity[0] - 1) / abs(((equity / equity.cummax()) - 1).min()),
        'volatility': returns.std() * np.sqrt(252 * 390) * 100,
    }
```

---

## 6. Walk-Forward Analysis

### Train/Test Split
```python
def walk_forward_analysis(data: pd.DataFrame, train_pct: float = 0.7):
    """Perform walk-forward analysis."""
    train_size = int(len(data) * train_pct)
    train = data[:train_size]
    test = data[train_size:]
    
    # Optimize on train
    bt_train = Backtest(train, OpenGateBacktest)
    optimal_params = bt_train.optimize(
        stop_buffer_ticks=[0.5, 1.0, 1.5, 2.0],
        maximize='Sharpe Ratio',
        return_optimization=True
    )
    
    # Apply to test
    bt_test = Backtest(test, OpenGateBacktest, 
                       strategy_params=optimal_params)
    test_results = bt_test.run()
    
    return {
        'train_stats': bt_train.results(),
        'test_stats': test_results,
        'params': optimal_params
    }
```

---

## 7. Monte Carlo Simulation

```python
def monte_carlo_simulation(trades: list, n_simulations: int = 1000):
    """Run Monte Carlo simulation on trade returns."""
    returns = np.array([t['return'] for t in trades])
    
    results = []
    for _ in range(n_simulations):
        shuffled = np.random.choice(returns, size=len(returns), replace=True)
        equity = (1 + shuffled / 100).cumprod()
        results.append(equity[-1])
    
    return {
        'median_return': np.median(results) - 1,
        'percentile_5': np.percentile(results, 5) - 1,
        'percentile_95': np.percentile(results, 95) - 1,
        'probability_of_ruin': sum(1 for r in results if r < 1) / len(results)
    }
```

---

## 8. Project Structure

```
open-gate-strategy/
├── pyproject.toml
├── uv.lock
├── ALGO.md
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── signals/
├── src/
│   ├── __init__.py
│   ├── strategy.py          # Core strategy class
│   ├── backtest.py          # Backtesting logic
│   ├── data_fetcher.py      # Data acquisition
│   ├── performance.py       # Metrics & analysis
│   └── visualization.py     # Plotting & charts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_strategy_development.ipynb
│   ├── 03_backtesting.ipynb
│   └── 04_optimization.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_strategy.py
│   └── test_backtest.py
└── reports/
    └── figures/
```

---

## 9. Implementation Checklist

- [ ] Set up `uv` project and install dependencies
- [ ] Implement `OpenGateStrategy` class
- [ ] Create data fetcher for yfinance / CSV import
- [ ] Build basic backtest loop
- [ ] Add confirmation logic (wick, engulfing, structure)
- [ ] Implement stop loss and take profit
- [ ] Calculate performance metrics
- [ ] Add visualization (equity curve, drawdown, trade markers)
- [ ] Run optimization on key parameters
- [ ] Perform walk-forward analysis
- [ ] Run Monte Carlo simulation
- [ ] Document edge cases and filters
- [ ] Write unit tests
- [ ] Validate against out-of-sample data

---

## 10. Common Pitfalls

1. **Overfitting** — Don't optimize too many parameters on limited data
2. **Lookahead bias** — Ensure gate is only set after 5-min candle closes
3. **Survivorship bias** — Use point-in-time data, not adjusted close
4. **Transaction costs** — Include commission + slippage in backtest
5. **Gap risk** — Account for overnight/weekend gaps in stop placement
6. **Session handling** — Properly reset state at market open

---

## 11. Key Libraries Reference

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `vectorbt` | Advanced backtesting & optimization |
| `backtesting` | Simple backtesting framework |
| `ta` | Technical indicators |
| `plotly` | Interactive visualization |
| `matplotlib` | Static plotting |
| `yfinance` | Free market data |
| `pyfolio` | Portfolio analytics |

---

## 12. Next Steps

1. Start with `uv init` and add dependencies
2. Implement the `OpenGateStrategy` class in `src/strategy.py`
3. Fetch historical 1-minute data for NQ or ES
4. Build a simple backtest loop first
5. Iterate with confirmation logic and parameter optimization
6. Move to vectorbt for production-grade analysis
