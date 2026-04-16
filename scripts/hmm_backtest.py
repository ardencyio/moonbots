"""
Historic HMM backtest runner.

At each out-of-sample bar T, the runner:
    1. Calls HMMEngine.predict_filtered(features[T]) — forward algorithm, no
       look-ahead (never calls model.predict).
    2. Asks StrategyOrchestrator.should_enter_position for the target exposure.
    3. Applies exposure = position_size * leverage to the bar T → T+1 return.

Usage:
    uv run python scripts/hmm_backtest.py --ticker SPY --start 2020-01-01 \\
        --end 2024-12-31 --interval 1d
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.core.hmm.feature_engineering import (  # noqa: E402
    compute_features,
    prepare_features_for_hmm,
)
from shared.core.hmm.hmm_engine import HMMEngine  # noqa: E402
from shared.core.hmm.regime_strategies import StrategyOrchestrator  # noqa: E402

BARS_PER_YEAR: dict[str, float] = {
    "1m": 98280.0,
    "5m": 19656.0,
    "15m": 6552.0,
    "1h": 1638.0,
    "1d": 252.0,
}


@dataclass
class BacktestResult:
    final_equity: float
    total_return_pct: float
    sharpe: float
    max_drawdown: float
    bar_count: int
    regime_distribution: dict[str, int]
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    exposures: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def run_walk_forward(
    df: pd.DataFrame,
    *,
    warmup_pct: float = 0.5,
    n_candidates: tuple[int, ...] = (3, 4, 5),
    stability_bars: int = 2,
    min_train_bars: int = 200,
    bars_per_year: float = 252.0,
    symbol: str = "SYMBOL",
) -> BacktestResult:
    if not 0.1 <= warmup_pct <= 0.9:
        raise ValueError(f"warmup_pct must be in [0.1, 0.9], got {warmup_pct}")

    df = df.copy()
    df.columns = df.columns.str.lower()
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    features = prepare_features_for_hmm(df)
    raw = compute_features(df)
    close = df["close"]
    ema50 = close.rolling(50).mean()

    if len(features) < 100:
        raise ValueError(f"Too few feature rows ({len(features)}); need >=100")

    split_idx = int(len(features) * warmup_pct)
    train_features = features.iloc[:split_idx]
    oos_features = features.iloc[split_idx:]

    if len(oos_features) < 20:
        raise ValueError(f"Too few OOS bars ({len(oos_features)}); need >=20")

    engine = HMMEngine(
        n_candidates=list(n_candidates),
        stability_bars=stability_bars,
        min_train_bars=min_train_bars,
    )
    engine.train(train_features)

    orchestrator = StrategyOrchestrator(n_regimes=engine.n_regimes)
    orchestrator.update_regime_infos(engine.regime_infos)

    equity = 1.0
    equity_points: dict[object, float] = {}
    exposure_points: dict[object, float] = {}
    regime_counter: Counter[str] = Counter()

    oos_idx = oos_features.index
    for i in range(len(oos_idx) - 1):
        bar_time = oos_idx[i]
        next_time = oos_idx[i + 1]

        feature_vec = oos_features.iloc[i].to_numpy(dtype=np.float64)
        ts = bar_time.to_pydatetime() if isinstance(bar_time, pd.Timestamp) else None
        regime_state = engine.predict_filtered(feature_vec, timestamp=ts)
        regime_counter[regime_state.label] += 1

        strategy = orchestrator.get_strategy_for_regime(
            regime_state.state_id, engine.regime_infos
        )
        current_price = float(close.loc[bar_time])
        vol = float(raw.loc[bar_time, "realized_vol"])
        trend = float(raw.loc[bar_time, "sma50_slope"])
        momentum = float(raw.loc[bar_time, "roc_10"])
        atr = float(raw.loc[bar_time, "atr_ratio"]) * current_price
        ema_val = ema50.loc[bar_time]
        ema = float(ema_val) if pd.notna(ema_val) else 0.0

        signal = orchestrator.should_enter_position(
            strategy=strategy,
            symbol=symbol,
            current_price=current_price,
            volatility=max(abs(vol), 1e-6),
            trend=trend,
            momentum=momentum,
            regime_strength=regime_state.probability,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            confidence=regime_state.probability,
            is_flickering=not regime_state.is_confirmed,
            atr=atr,
            ema50=ema,
            timestamp=ts,
        )

        exposure = (
            signal.position_size_pct * signal.leverage
            if signal.direction == "long"
            else 0.0
        )
        bar_return = float(close.loc[next_time] / close.loc[bar_time] - 1.0)
        equity *= 1.0 + exposure * bar_return

        equity_points[bar_time] = equity
        exposure_points[bar_time] = exposure

    eq_series = pd.Series(equity_points, dtype=float)
    exp_series = pd.Series(exposure_points, dtype=float)

    returns = eq_series.pct_change().dropna()
    if len(returns) > 1 and float(returns.std()) > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    if len(eq_series):
        running_max = eq_series.cummax()
        max_drawdown = float((eq_series / running_max - 1.0).min())
    else:
        max_drawdown = 0.0

    return BacktestResult(
        final_equity=float(equity),
        total_return_pct=float((equity - 1.0) * 100.0),
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        bar_count=len(eq_series),
        regime_distribution=dict(regime_counter),
        equity_curve=eq_series,
        exposures=exp_series,
    )


def _fetch_ohlcv(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    from backtest.data.yfinance_fetcher import YFinanceFetcher

    fetcher = YFinanceFetcher()
    df = fetcher.fetch(ticker, start=start, end=end, resolution=interval)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} {start}..{end} @ {interval}")
    df = df.rename(columns=str.lower)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Fetched frame missing columns: {sorted(missing)}")
    return df[["open", "high", "low", "close", "volume"]]


def _print_summary(ticker: str, result: BacktestResult) -> None:
    print(f"=== HMM Backtest: {ticker} ===")
    print(f"Bar count:          {result.bar_count}")
    print(f"Final equity:       {result.final_equity:.4f}")
    print(f"Total return:       {result.total_return_pct:+.2f}%")
    print(f"Sharpe ratio:       {result.sharpe:+.3f}")
    print(f"Max drawdown:       {result.max_drawdown * 100:+.2f}%")
    print("Regime distribution:")
    total = sum(result.regime_distribution.values()) or 1
    for label, count in sorted(
        result.regime_distribution.items(), key=lambda kv: -kv[1]
    ):
        print(f"  {label:<16} {count:>6} ({count / total * 100:5.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="HMM walk-forward backtest runner")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--interval", default="1d", choices=sorted(BARS_PER_YEAR))
    parser.add_argument("--warmup-pct", type=float, default=0.5)
    args = parser.parse_args()

    df = _fetch_ohlcv(args.ticker, args.start, args.end, args.interval)
    result = run_walk_forward(
        df,
        warmup_pct=args.warmup_pct,
        bars_per_year=BARS_PER_YEAR[args.interval],
        symbol=args.ticker,
    )
    _print_summary(args.ticker, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
