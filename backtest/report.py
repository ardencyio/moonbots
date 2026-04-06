"""Performance report generation for backtest results."""

from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_report(result: dict, output_dir: str = "reports") -> Path:
    """Generate a performance report from backtest results.

    Args:
        result: dict from BacktestRunner.run()
        output_dir: directory to write reports to

    Returns:
        Path to the generated report file
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out / f"backtest_report_{timestamp}.txt"

    lines = [
        "=" * 60,
        "BACKTEST PERFORMANCE REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        "",
        f"Total Trades:       {metrics['total_trades']}",
        f"Win Rate:           {metrics['win_rate']:.2%}",
        f"Profit Factor:      {metrics['profit_factor']:.2f}",
        f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}",
        f"Max Drawdown:       {metrics['max_drawdown']:.2%}",
        f"Expectancy:         ${metrics['expectancy']:.2f}",
        f"Calmar Ratio:       {metrics['calmar_ratio']:.2f}",
        f"Total Return:       {metrics['total_return']:.2%}",
        "",
        "=" * 60,
    ]

    report_path.write_text("\n".join(lines))

    # Equity curve plot
    equity = result.get("equity_curve")
    if equity is not None and len(equity) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        equity.plot(ax=ax, title="Equity Curve")
        ax.set_ylabel("Equity ($)")
        fig_path = out / f"equity_curve_{timestamp}.png"
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    return report_path
