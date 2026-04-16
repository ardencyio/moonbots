"""Tests for scripts/hmm_backtest.py — walk-forward runner on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.hmm_backtest import BacktestResult, run_walk_forward
from shared.core.hmm.hmm_engine import HMMEngine


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    np.random.seed(7)
    n_bars = 1200
    returns = np.random.randn(n_bars) * 0.015
    regime_shift = np.concatenate(
        [
            np.random.randn(400) * 0.008 + 0.0005,
            np.random.randn(400) * 0.025 - 0.0010,
            np.random.randn(400) * 0.015 + 0.0002,
        ]
    )
    close = np.cumprod(1 + returns + regime_shift) * 100.0
    open_ = close * (1 + np.random.randn(n_bars) * 0.002)
    high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_bars) * 0.004))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_bars) * 0.004))
    volume = np.abs(np.random.randn(n_bars) * 1e6 + 1e7)
    index = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


class TestRunWalkForward:
    def test_returns_well_formed_result(self, synthetic_ohlcv):
        result = run_walk_forward(
            synthetic_ohlcv,
            warmup_pct=0.5,
            n_candidates=(3,),
            stability_bars=1,
            min_train_bars=150,
        )
        assert isinstance(result, BacktestResult)
        assert result.bar_count > 0
        assert np.isfinite(result.final_equity)
        assert result.final_equity > 0.0
        assert np.isfinite(result.sharpe)
        assert result.max_drawdown <= 0.0
        assert sum(result.regime_distribution.values()) == result.bar_count
        assert len(result.equity_curve) == result.bar_count

    def test_rejects_invalid_warmup_pct(self, synthetic_ohlcv):
        with pytest.raises(ValueError, match="warmup_pct"):
            run_walk_forward(synthetic_ohlcv, warmup_pct=0.05)

    def test_rejects_missing_ohlcv_columns(self):
        bad = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            run_walk_forward(bad)

    def test_no_lookahead_uses_predict_filtered_not_predict(
        self, synthetic_ohlcv, monkeypatch
    ):
        def blow_up(self, *args, **kwargs):
            raise AssertionError(
                "Backtest must not call HMMEngine.predict (Viterbi) — look-ahead bias"
            )

        monkeypatch.setattr(HMMEngine, "predict", blow_up, raising=False)
        result = run_walk_forward(
            synthetic_ohlcv,
            warmup_pct=0.5,
            n_candidates=(3,),
            stability_bars=1,
            min_train_bars=150,
        )
        assert result.bar_count > 0
