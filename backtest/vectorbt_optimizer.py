"""Vectorbt integration for fast parameter optimization."""

from typing import Optional
import numpy as np
import pandas as pd

import vectorbt as vbt
from vectorbt.portfolio import Portfolio


class VectorbtOptimizer:
    """Vectorbt-based optimizer for fast parameter sweeps."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.price = data["Close"]

    def run_vectorbt_sweep(
        self,
        param_grid: dict,
        entries: Optional[pd.DataFrame] = None,
        exits: Optional[pd.DataFrame] = None,
    ) -> Portfolio:
        """Run vectorized backtest across multiple parameter combinations.

        Args:
            param_grid: Dict of parameter name -> list of values to test
            entries: DataFrame of entry signals (if pre-calculated)
            exits: DataFrame of exit signals (if pre-calculated)

        Returns:
            Portfolio with results for all parameter combinations
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        results = []
        for combo in np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names)):
            params = dict(zip(param_names, combo))

            # Generate signals using the OpenGate strategy
            if entries is None or exits is None:
                entries, exits = self._generate_signals(params)

            # Run portfolio simulation for this combination
            pf = vbt.Portfolio.from_signals(
                close=self.price,
                entries=entries,
                exits=exits,
                size=params.get("position_size", 1),
                fees=params.get("fees", 0.001),
                freq="1T",  # 1 minute bars
            )

            results.append({
                **params,
                "sharpe": pf.sharpe_ratio(),
                "total_return": pf.total_return(),
                "max_drawdown": pf.max_drawdown(),
                "win_rate": pf.win_rate(),
                "profit_factor": pf.profit_factor(),
            })

        return pd.DataFrame(results)

    def _generate_signals(self, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate entry/exit signals using vectorbt indicators."""
        # Use params for future signal customization if needed

        # Use SMA for signal generation (can be replaced with custom logic)
        fast_ma = vbt.MA.run(self.price, window=10, short_name="fast")
        slow_ma = vbt.MA.run(self.price, window=50, short_name="slow")

        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)

        return entries, exits

    def run_fast_sweep(self, param_grid: dict, n_combos: int = 100) -> pd.DataFrame:
        """Run fast parameter sweep with random sampling.

        Uses vectorized operations for speed - test 1000+ combos in seconds.
        """
        results = []

        for _ in range(n_combos):
            # Random parameter sampling
            combo = {name: np.random.choice(values) for name, values in param_grid.items()}

            # Generate signals
            fast_window = combo.get("fast_window", 10)
            slow_window = combo.get("slow_window", 50)

            fast_ma = vbt.MA.run(self.price, window=fast_window)
            slow_ma = vbt.MA.run(self.price, window=slow_window)

            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)

            pf = vbt.Portfolio.from_signals(
                close=self.price,
                entries=entries,
                exits=exits,
                size=1,
                fees=0.001,
                freq="1T",
            )

            results.append({
                **combo,
                "sharpe": pf.sharpe_ratio(),
                "total_return": pf.total_return(),
                "max_drawdown": pf.max_drawdown(),
                "win_rate": pf.win_rate(),
            })

        return pd.DataFrame(results)
