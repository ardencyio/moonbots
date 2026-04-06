"""Backtesting.py integration for strategy backtesting."""

from typing import Optional

import pandas as pd

from backtesting import Backtest, Strategy


class OpenGateBacktestStrategy(Strategy):
    """Adapter for OpenGateStrategy to work with backtesting.py."""

    # Strategy parameters (will be set by backtesting.py optimizer)
    gate_minutes = 5
    stop_buffer = 1.0
    min_risk_reward = 2.0

    def init(self):
        """Initialize strategy indicators."""
        from backtest.strategies.open_gate import OpenGateStrategy

        config = {
            "gate_candle_minutes": self.gate_minutes,
            "stop_buffer_ticks": self.stop_buffer,
            "min_risk_reward": self.min_risk_reward,
            "use_market_hours": False,  # Use first N candles of each day
        }
        self.strategy = OpenGateStrategy(config)

        # Initialize gate using first N candles of first day
        # Get the first gate_minutes candles from the data
        n_candles = min(self.gate_minutes, len(self.data))
        gate_high = float(self.data.High[:n_candles].max())
        gate_low = float(self.data.Low[:n_candles].min())

        self.strategy.gate_high = gate_high
        self.strategy.gate_low = gate_low
        self.strategy.gate_set = True
        self.strategy.broke_above = False
        self.strategy.broke_below = False

    def next(self):
        """Process each bar in the dataset."""
        # Get the current row as a dict-like object
        current_row = pd.Series({
            "Open": self.data.Open[-1],
            "High": self.data.High[-1],
            "Low": self.data.Low[-1],
            "Close": self.data.Close[-1],
            "Volume": self.data.Volume[-1] if hasattr(self.data, "Volume") else 0,
        })

        # Check for exit signal first (if position is open)
        if self.position:
            self.strategy.position_open = True  # sync flag
            # Check stop loss / take profit
            exit_signal = self.strategy.check_exit(current_row)
            if exit_signal:
                self.position.close()
                return
        else:
            self.strategy.position_open = False  # ensure flag is synced when flat

        # Generate new signal
        signal = self.strategy.on_data(current_row)

        # Check if we have an entry signal
        if signal and signal.get("signal_type") == "entry":
            direction = signal.get("direction")
            size = signal.get("size", 1)

            if direction == "long":
                self.buy(size=size)
            elif direction == "short":
                self.sell(size=size)


class BacktestingPyAdapter:
    """Adapter to run backtesting with backtesting.py library."""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100_000):
        self.data = data
        self.initial_capital = initial_capital

    def run(
        self,
        strategy_class: type = OpenGateBacktestStrategy,
        **kwargs,
    ) -> dict:
        """Run backtest using backtesting.py."""
        bt = Backtest(
            self.data,
            strategy_class,
            cash=self.initial_capital,
            commission=0.001,  # 0.1% commission
            exclusive_orders=True,
            **kwargs,
        )

        stats = bt.run()
        return {
            "stats": stats,
            "strategy": bt.strategy,
        }

    def plot(self, bt_result: dict, filename: Optional[str] = None) -> None:
        """Generate interactive plot of backtest results."""
        import os

        os.environ["BACKTESTING_PLOTLY_RENDERER"] = "png"

        bt = bt_result["stats"]
        if filename:
            bt.plot(filename=filename)
        else:
            bt.plot()

    def optimize(
        self,
        param_grid: dict,
        maximize: str = "Sharpe Ratio",
        **kwargs,
    ) -> dict:
        """Optimize strategy parameters using grid search."""
        bt = Backtest(
            self.data,
            OpenGateBacktestStrategy,
            cash=self.initial_capital,
            commission=0.001,
            exclusive_orders=True,
        )

        stats = bt.optimize(
            gate_minutes=param_grid.get("gate_minutes", range(3, 8)),
            stop_buffer=param_grid.get("stop_buffer", [0.5, 1.0, 1.5, 2.0]),
            min_risk_reward=param_grid.get("min_risk_reward", [1.5, 2.0, 2.5]),
            maximize=maximize,
            **kwargs,
        )

        return {"stats": stats, "strategy": bt.strategy}
