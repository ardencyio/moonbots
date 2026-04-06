"""Example backtest using Polygon.io data for NQ=F.

This script demonstrates running a backtest with the Open Gate strategy
using Polygon.io API data. Requires POLYGON_API_KEY environment variable.

Usage:
    uv run python backtest/examples/polygon_backtest.py [--days 365]

Or with varlock:
    varlock run -- uv run python backtest/examples/polygon_backtest.py --days 365
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtest.backtest import BacktestRunner
from backtest.strategies.open_gate import OpenGateStrategy
from backtest.data.polygon_fetcher import PolygonFetcher
from scripts.env_loader import get_env


def fetch_nq_data_from_polygon(days: int = 365) -> pd.DataFrame:
    """Fetch NQ=F data from Polygon.io API.

    Args:
        days: Number of days of data to fetch

    Returns:
        OHLCV DataFrame
    """
    # Get API key from env (varlock or .env)
    polygon_key = get_env("POLYGON_API_KEY")
    if not polygon_key:
        raise RuntimeError(
            "POLYGON_API_KEY not found. "
            "Set it in your .env file or inject via varlock."
        )

    fetcher = PolygonFetcher(polygon_key)
    end = datetime.now()
    start = end - timedelta(days=days)

    # Use 5-minute data for better backtest granularity
    data = fetcher.fetch("NQ=F", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5m")

    if data.empty:
        raise RuntimeError("Failed to fetch NQ=F data from Polygon.io")

    print(f"Fetched {len(data)} rows of 5-minute data from Polygon.io")
    return data


def run_polygon_backtest(days: int = 365):
    """Run an example backtest on NQ=F using Polygon data."""
    print("=" * 60)
    print("Open Gate Strategy Backtest - NQ=F (Polygon.io)")
    print("=" * 60)

    # Fetch data
    print(f"\nFetching {days} days of data from Polygon.io API...")
    data = fetch_nq_data_from_polygon(days=days)
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Rows: {len(data)}")

    # Configure strategy
    config = {
        "gate_candle_minutes": 5,
        "stop_buffer_ticks": 1.0,
        "min_risk_reward": 2.0,
        "use_market_hours": False,
    }
    strategy = OpenGateStrategy(config)

    # Run backtest
    print("\nRunning backtest...")
    runner = BacktestRunner(strategy, initial_capital=50_000)
    result = runner.run(data)

    metrics = result["metrics"]
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Trades:       {metrics['total_trades']}")
    print(f"Win Rate:           {metrics['win_rate']:.2%}")
    print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown']:.2%}")
    print(f"Expectancy:         ${metrics['expectancy']:.2f}")
    print(f"Calmar Ratio:       {metrics['calmar_ratio']:.2f}")
    print(f"Total Return:       {metrics['total_return']:.2%}")
    print("=" * 60)

    # Validate against deployment thresholds
    print("\nDeployment Threshold Validation:")
    print("-" * 40)
    checks = [
        ("Sharpe Ratio > 1.0", metrics['sharpe_ratio'] > 1.0),
        ("Max Drawdown < 15%", abs(metrics['max_drawdown']) < 0.15),
        ("Win Rate > 45%", metrics['win_rate'] > 0.45),
        ("Profit Factor > 1.3", metrics['profit_factor'] > 1.3),
    ]

    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")

    # Generate report
    from backtest.report import generate_report
    report_path = generate_report(result, output_dir="reports")
    print(f"\nReport saved to: {report_path}")

    # Plot equity curve
    print("\nGenerating equity curve plot...")
    runner.plot(filename="reports/equity_curve_polygon.png")
    print("Plot saved to: reports/equity_curve_polygon.png")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Open Gate strategy backtest with Polygon.io data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (Polygon limit: ~30 days for 5m)",
    )
    args = parser.parse_args()

    try:
        result = run_polygon_backtest(days=args.days)
        print("\nBacktest completed successfully!")
    except Exception as e:
        print(f"\nBacktest failed: {e}")
        import traceback
        traceback.print_exc()
