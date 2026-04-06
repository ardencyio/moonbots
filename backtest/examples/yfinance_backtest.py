"""Example backtest using yfinance or Polygon.io data for NQ=F.

This script demonstrates running a backtest with the Open Gate strategy
using free yfinance data or Polygon.io API data (requires API key).

Usage:
    uv run python backtest/examples/yfinance_backtest.py --source yfinance
    uv run python backtest/examples/yfinance_backtest.py --source polygon
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from backtest.backtest import BacktestRunner
from backtest.strategies.open_gate import OpenGateStrategy
from backtest.data.yfinance_fetcher import YFinanceFetcher
from scripts.env_loader import get_env


def fetch_nq_data_from_polygon(days: int = 365) -> pd.DataFrame:
    """Fetch NQ=F data from Polygon.io API.

    Args:
        days: Number of days of data to fetch

    Returns:
        OHLCV DataFrame
    """
    polygon_key = get_env("POLYGON_API_KEY")
    if not polygon_key:
        raise RuntimeError("POLYGON_API_KEY not found")

    from backtest.data.polygon_fetcher import PolygonFetcher
    fetcher = PolygonFetcher(polygon_key)
    end = datetime.now()
    start = end - timedelta(days=days)

    data = fetcher.fetch("NQ=F", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5m")

    if data.empty:
        raise RuntimeError("Failed to fetch NQ=F data from Polygon.io")

    print(f"Fetched {len(data)} rows of 5-minute data from Polygon.io")
    return data


def fetch_nq_data(days: int = 730) -> pd.DataFrame:
    """Fetch NQ=F data from yfinance.

    Args:
        days: Number of days of data to fetch (max ~1 year of 1m data available)

    Returns:
        OHLCV DataFrame
    """
    fetcher = YFinanceFetcher()
    end = datetime.now()
    start = end - timedelta(days=days)

    # Try 1-minute data first (limited to ~8 days)
    try:
        data = fetcher.fetch("NQ=F", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1m")
        if not data.empty:
            print(f"Fetched {len(data)} rows of 1-minute data")
            return data
    except Exception as e:
        print(f"1-minute fetch failed: {e}")

    # Fall back to 5-minute data
    try:
        data = fetcher.fetch("NQ=F", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5m")
        if not data.empty:
            print(f"Fetched {len(data)} rows of 5-minute data")
            return data
    except Exception as e:
        print(f"5-minute fetch failed: {e}")

    # Fall back to daily data
    try:
        data = fetcher.fetch("NQ=F", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1d")
        if not data.empty:
            print(f"Fetched {len(data)} rows of daily data")
            return data
    except Exception as e:
        print(f"Daily fetch failed: {e}")

    raise RuntimeError("Failed to fetch NQ=F data from yfinance")


def run_example_backtest():
    """Run an example backtest on NQ=F."""
    print("=" * 60)
    print("Open Gate Strategy Backtest - NQ=F")
    print("=" * 60)

    # Fetch data
    print("\nFetching data from yfinance (no API key required)...")
    data = fetch_nq_data(days=365)
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Rows: {len(data)}")

    # Configure strategy
    config = {
        "gate_candle_minutes": 5,
        "stop_buffer_ticks": 1.0,
        "min_risk_reward": 2.0,
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
    runner.plot(filename="reports/equity_curve.png")
    print("Plot saved to: reports/equity_curve.png")

    # Walk-forward analysis
    print("\nRunning walk-forward analysis...")
    wf_result = runner.walk_forward(data, train_pct=0.7)
    print(f"Train metrics: Sharpe={wf_result['train']['metrics']['sharpe_ratio']:.2f}")
    print(f"Test metrics:  Sharpe={wf_result['test']['metrics']['sharpe_ratio']:.2f}")

    # Monte Carlo simulation
    if metrics['total_trades'] > 0:
        print("\nRunning Monte Carlo simulation (1000 iterations)...")
        mc_result = runner.monte_carlo(result["trades"], n_simulations=1000)
        print(f"Median return:    {mc_result['median_return']:.2%}")
        print(f"5th percentile:   {mc_result['pct_5']:.2%}")
        print(f"95th percentile:  {mc_result['pct_95']:.2%}")
        print(f"Probability of ruin: {mc_result['ruin_prob']:.2%}")

    return result


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run Open Gate strategy backtest with yfinance or Polygon.io data"
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "polygon"],
        default="yfinance",
        help="Data source: yfinance (free) or polygon (requires API key)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch",
    )
    args = parser.parse_args()

    if args.source == "yfinance":
        print("=" * 60)
        print("Open Gate Strategy Backtest - NQ=F (yfinance)")
        print("=" * 60)
        print("\nFetching data from yfinance (no API key required)...")
        data = fetch_nq_data(days=args.days)
    else:
        print("=" * 60)
        print("Open Gate Strategy Backtest - NQ=F (Polygon.io)")
        print("=" * 60)
        print("\nFetching data from Polygon.io API...")
        data = fetch_nq_data_from_polygon(days=args.days)

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
    runner.plot(filename=f"reports/equity_curve_{args.source}.png")
    print(f"Plot saved to: reports/equity_curve_{args.source}.png")

    # Walk-forward analysis
    print("\nRunning walk-forward analysis...")
    wf_result = runner.walk_forward(data, train_pct=0.7)
    print(f"Train metrics: Sharpe={wf_result['train']['metrics']['sharpe_ratio']:.2f}")
    print(f"Test metrics:  Sharpe={wf_result['test']['metrics']['sharpe_ratio']:.2f}")

    # Monte Carlo simulation
    if metrics['total_trades'] > 0:
        print("\nRunning Monte Carlo simulation (1000 iterations)...")
        mc_result = runner.monte_carlo(result["trades"], n_simulations=1000)
        print(f"Median return:    {mc_result['median_return']:.2%}")
        print(f"5th percentile:   {mc_result['pct_5']:.2%}")
        print(f"95th percentile:  {mc_result['pct_95']:.2%}")
        print(f"Probability of ruin: {mc_result['ruin_prob']:.2%}")


if __name__ == "__main__":
    try:
        main()
        print("\n✅ Backtest completed successfully!")
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
