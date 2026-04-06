"""Backtesting framework using backtesting.py for strategy testing."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtesting import Backtest

from backtest.backtesting_runner import OpenGateBacktestStrategy


class BacktestRunner:
    """Run backtests on strategies using backtesting.py library.

    Uses the backtesting.py library's Backtest class for event-driven
    simulation, then calculates custom metrics.
    """

    def __init__(self, strategy, initial_capital: float = 100_000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.bt = None
        self.result = None

    def load_data(self, path: str) -> pd.DataFrame:
        """Load historical data from CSV or parquet."""
        p = Path(path)
        if p.suffix == ".csv":
            return pd.read_csv(p, index_col=0, parse_dates=True)
        return pd.read_parquet(p)

    def run(self, data: pd.DataFrame) -> dict:
        """Run backtest using backtesting.py."""
        # Convert strategy to backtesting.py format
        bt = Backtest(
            data,
            OpenGateBacktestStrategy,
            cash=self.initial_capital,
            commission=0.001,
            exclusive_orders=True,
        )

        self.bt = bt
        self.result = bt.run()

        # Extract trades and equity curve
        trades = self._extract_trades()
        equity_series = self._extract_equity()

        metrics = self._calculate_metrics(equity_series, trades)
        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_series,
            "returns": equity_series.pct_change().dropna(),
            "stats": self.result,
        }

    def _extract_trades(self) -> list:
        """Extract trades from backtesting.py result."""
        if self.result is None:
            return []

        # Access the trades DataFrame from the result (_trades is the internal attribute)
        trades_df = getattr(self.result, "_trades", None)
        if trades_df is None or len(trades_df) == 0:
            return []

        trades = []
        for _, row in trades_df.iterrows():
            pnl = row.get("PnL", 0)
            trades.append({
                "timestamp": row.get("EntryTime", row.get("ExitTime")),
                "direction": "long" if row.get("Size", 0) > 0 else "short",
                "entry_price": row.get("EntryPrice", 0),
                "exit_price": row.get("ExitPrice", 0),
                "pnl": float(pnl) if pnl else 0,
                "reason": row.get("Tag", "trade_closed"),
            })

        return trades

    def _extract_equity(self) -> pd.Series:
        """Extract equity curve from backtesting.py result."""
        if self.result is None:
            return pd.Series([self.initial_capital])

        # _equity_curve is the internal attribute name
        equity_curve = getattr(self.result, "_equity_curve", None)
        if equity_curve is not None:
            return equity_curve["Equity"]

        return pd.Series([self.initial_capital])

    def _calculate_metrics(self, equity: pd.Series, trades: list) -> dict:
        """Calculate performance metrics."""
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "expectancy": 0,
                "calmar_ratio": 0,
                "total_return": 0,
                "avg_trade_duration": 0,
            }

        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 1 else 0
        max_dd = ((equity / equity.cummax()) - 1).min() if len(equity) > 1 else 0

        returns = equity.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 390) if returns.std() > 0 else 0
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        expectancy = np.mean(pnls) if pnls else 0

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "expectancy": expectancy,
            "calmar_ratio": calmar,
            "total_return": total_return,
            "avg_trade_duration": 0,
        }

    def plot(self, filename: Optional[str] = None) -> None:
        """Generate plot of backtest results using backtesting.py."""
        if self.bt is None:
            raise RuntimeError("Run backtest first with .run()")
        if filename:
            self.bt.plot(filename=filename)
        else:
            self.bt.plot()

    def walk_forward(self, data: pd.DataFrame, train_pct: float = 0.7):
        """Walk-forward analysis with train/test split."""
        train_size = int(len(data) * train_pct)
        train = data.iloc[:train_size]
        test = data.iloc[train_size:]

        train_result = self.run(train)
        test_result = self.run(test)

        return {
            "train": train_result,
            "test": test_result,
            "train_size": len(train),
            "test_size": len(test),
        }

    def monte_carlo(self, trades: list, n_simulations: int = 1000):
        """Monte Carlo simulation on trade returns."""
        pnls = np.array([t.get("pnl", 0) for t in trades])
        if len(pnls) == 0:
            return {"median_return": 0, "pct_5": 0, "pct_95": 0, "ruin_prob": 0}

        results = []
        for _ in range(n_simulations):
            shuffled = np.random.choice(pnls, size=len(pnls), replace=True)
            equity = (1 + shuffled / self.initial_capital).cumprod()
            results.append(equity[-1])

        return {
            "median_return": float(np.median(results) - 1),
            "pct_5": float(np.percentile(results, 5) - 1),
            "pct_95": float(np.percentile(results, 95) - 1),
            "ruin_prob": float(sum(1 for r in results if r < 1) / len(results)),
        }
