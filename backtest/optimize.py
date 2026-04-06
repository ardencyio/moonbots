"""Parameter optimization for trading strategies.

Supports both backtesting.py (detailed) and vectorbt (fast) for parameter sweeps.
"""

from itertools import product
from typing import Callable, Optional
import pandas as pd
import numpy as np

import vectorbt as vbt


def grid_search_backtesting(
    strategy_factory: Callable,
    data: pd.DataFrame,
    param_grid: dict,
    maximize: str = "Sharpe Ratio",
) -> pd.DataFrame:
    """Grid search over parameter combinations using backtesting.py.

    Args:
        strategy_factory: callable that accepts a dict of params and returns a strategy instance
        data: historical OHLCV DataFrame
        param_grid: dict of param_name -> list of values
        maximize: metric to maximize

    Returns:
        DataFrame with results sorted by sharpe ratio
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        strategy = strategy_factory(params)
        try:
            from backtest.backtest import BacktestRunner
            runner = BacktestRunner(strategy)
            result = runner.run(data)
            metrics = result["metrics"]
            results.append({
                **params,
                "sharpe": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "total_return": metrics["total_return"],
                "total_trades": metrics["total_trades"],
            })
        except Exception as e:
            results.append({
                **params,
                "sharpe": float("-inf"),
                "error": str(e),
            })

    df = pd.DataFrame(results)
    return df.sort_values("sharpe", ascending=False)


def grid_search_vectorbt(
    data: pd.DataFrame,
    param_grid: dict,
    n_combos: int = 100,
) -> pd.DataFrame:
    """Fast parameter sweep using vectorbt.

    Uses vectorized operations to test 1000+ parameter combinations in seconds.
    Ideal for initial parameter exploration.

    Args:
        data: historical OHLCV DataFrame
        param_grid: dict of param_name -> list of values
        n_combos: number of random combinations to test

    Returns:
        DataFrame with results sorted by sharpe ratio
    """
    price = data["Close"]
    results = []

    for _ in range(n_combos):
        # Random parameter sampling
        combo = {}
        for name, values in param_grid.items():
            if isinstance(values, list):
                combo[name] = np.random.choice(values)
            elif isinstance(values, range):
                combo[name] = np.random.choice(list(values))
            else:
                combo[name] = values

        # Generate signals using SMA crossover
        fast_window = combo.get("fast_window", 10)
        slow_window = combo.get("slow_window", 50)

        try:
            fast_ma = vbt.MA.run(price, window=fast_window)
            slow_ma = vbt.MA.run(price, window=slow_window)

            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)

            pf = vbt.Portfolio.from_signals(
                close=price,
                entries=entries,
                exits=exits,
                size=1,
                fees=combo.get("fees", 0.001),
                freq="1T",
            )

            results.append({
                **combo,
                "fast_window": fast_window,
                "slow_window": slow_window,
                "sharpe": float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0,
                "total_return": float(pf.total_return()),
                "max_drawdown": float(pf.max_drawdown()),
                "win_rate": float(pf.win_rate()),
                "profit_factor": float(pf.profit_factor()) if not np.isnan(pf.profit_factor()) else 0,
            })
        except Exception as e:
            results.append({
                **combo,
                "sharpe": float("-inf"),
                "error": str(e),
            })

    df = pd.DataFrame(results)
    return df.sort_values("sharpe", ascending=False)


def hybrid_optimize(
    data: pd.DataFrame,
    fast_param_grid: dict,
    detailed_param_grid: dict,
    n_fast_combos: int = 100,
    strategy_factory: Optional[Callable] = None,
) -> pd.DataFrame:
    """Hybrid optimization: fast vectorbt sweep followed by detailed backtesting.py.

    Args:
        data: historical OHLCV DataFrame
        fast_param_grid: parameters for fast vectorbt sweep
        detailed_param_grid: parameters for detailed backtesting.py
        n_fast_combos: number of combinations to test in fast phase
        strategy_factory: factory function for creating strategy instances

    Returns:
        DataFrame with all results
    """
    # Phase 1: Fast vectorbt sweep
    fast_results = grid_search_vectorbt(data, fast_param_grid, n_fast_combos)

    # Get top performers
    top_n = min(20, len(fast_results))
    top_params = fast_results.head(top_n)

    # Phase 2: Detailed backtesting.py on top candidates
    detailed_results = []
    for _, row in top_params.iterrows():
        params = {
            "fast_window": int(row.get("fast_window", 10)),
            "slow_window": int(row.get("slow_window", 50)),
        }
        if strategy_factory:
            strategy = strategy_factory(params)
        else:
            strategy = None  # Use default

        # Use backtesting.py with params
        try:
            from backtest.backtest import BacktestRunner
            runner = BacktestRunner(strategy)
            result = runner.run(data)
            metrics = result["metrics"]
            detailed_results.append({
                **params,
                "sharpe": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "total_return": metrics["total_return"],
                "total_trades": metrics["total_trades"],
            })
        except Exception as e:
            detailed_results.append({
                **params,
                "sharpe": float("-inf"),
                "error": str(e),
            })

    detailed_df = pd.DataFrame(detailed_results)
    return pd.concat([fast_results, detailed_df]).sort_values("sharpe", ascending=False)
