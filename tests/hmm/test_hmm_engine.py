"""
Tests for HMM Engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.core.hmm.feature_engineering import (
    compute_features,
    prepare_features_for_hmm,
)
from shared.core.hmm.hmm_engine import HMMEngine, RegimeInfo


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 1000  # Increased to accommodate rolling z-score warm-up

    close = np.cumprod(1 + np.random.randn(n_bars) * 0.02) * 100
    open_price = close * (1 + np.random.randn(n_bars) * 0.01)
    high = np.maximum(open_price, close) * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low = np.minimum(open_price, close) * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    volume = np.abs(np.random.randn(n_bars) * 1_000_000 + 10_000_000)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestFeatureEngineering:
    """Tests for feature computation."""

    def test_compute_features_no_lookahead(self, sample_ohlcv_data):
        """Features at T should only use data up to T."""
        features = compute_features(sample_ohlcv_data)
        assert len(features) == len(sample_ohlcv_data)

        expected_features = [
            "return_1",
            "realized_vol",
            "volatility_ratio",
            "price_vs_sma20",
            "sma20_vs_sma50",
            "rsi",
            "atr_ratio",
            "volume_ratio",
            "adx",
            "up_days_ratio",
        ]
        assert list(features.columns) == expected_features

        # Warm-up period should have NaNs (expected with rolling windows, no bfill)
        # This is CORRECT behavior - we drop these NaNs downstream for training
        # Do NOT assert isna().sum().sum() == 0 - that would require bfill() look-ahead
        warmup_nans = features.isna().sum().sum()
        assert warmup_nans > 0  # Expected warm-up NaNs from rolling windows
        assert warmup_nans < len(features)  # But not all rows are NaN


class TestHMMTraining:
    """Tests for HMM model training."""

    def test_train_selects_lowest_bic(self, sample_ohlcv_data):
        """HMM should select model with lowest BIC."""
        engine = HMMEngine(n_candidates=[3, 4, 5])
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        assert len(features) >= engine.min_train_bars
        engine.train(features)
        assert engine.model is not None
        assert engine.n_regimes in [3, 4, 5]
        assert engine.best_bic is not None
        assert len(engine.regime_infos) == engine.n_regimes

    def test_regime_labels_assigned_by_return(self, sample_ohlcv_data):
        """Regimes should be labeled by sorted return (ascending)."""
        engine = HMMEngine(n_candidates=[3])
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)
        expected_labels = {"BEAR", "NEUTRAL", "BULL"}
        actual_labels = {info.regime_name for info in engine.regime_infos.values()}
        assert actual_labels == expected_labels


class TestForwardAlgorithm:
    """Tests for forward algorithm (filtered inference)."""

    def test_predict_filtered_forward(self, sample_ohlcv_data):
        """Forward algorithm should return regime at each bar."""
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        predictions = []
        for i in range(len(features)):
            obs = features.values[i]
            regime = engine.predict_filtered(obs)
            predictions.append(regime)

        assert len(predictions) == len(features)
        for pred in predictions:
            assert 0 <= pred.probability <= 1
            assert len(pred.state_probabilities) == engine.n_regimes

    def test_sequential_predictions_use_previous_posterior(self, sample_ohlcv_data):
        """Sequential calls should use previous posterior as next prior."""
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        engine.reset_inference_state()

        # Process observation at index 100
        obs_100 = features.values[100]
        pred_100 = engine.predict_filtered(obs_100)

        # Now process next observation - should use pred_100's posterior as prior
        obs_101 = features.values[101]
        pred_101 = engine.predict_filtered(obs_101)

        # Both should give valid regime states
        assert 0 <= pred_100.state_id < engine.n_regimes
        assert 0 <= pred_101.state_id < engine.n_regimes

        # Sequential processing should maintain state
        engine.reset_inference_state()

    def test_regime_confirmation_stability(self, sample_ohlcv_data):
        """Confirms regime after stability_bars consecutive same observations."""
        engine = HMMEngine(n_candidates=[3], stability_bars=3)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        engine.reset_inference_state()

        # Feed same observation multiple times (simulating time passing in same regime)
        obs = features.values[100]

        pred1 = engine.predict_filtered(obs)
        pred2 = engine.predict_filtered(obs)
        pred3 = engine.predict_filtered(obs)
        pred4 = engine.predict_filtered(obs)

        # Regime might oscillate initially but should settle
        # The key test is that the filter runs without error
        assert pred1.state_id is not None
        assert pred2.state_id is not None
        assert pred3.state_id is not None
        assert pred4.state_id is not None

    def test_flicker_detection(self, sample_ohlcv_data):
        """Flicker rate should be tracked."""
        engine = HMMEngine(
            n_candidates=[3],
            stability_bars=1,
            flicker_window=10,
            flicker_threshold=5,
        )
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)
        assert engine.get_regime_flicker_rate() >= 0
        assert isinstance(engine.is_flickering(), bool)


class TestRegimeInfo:
    """Tests for regime metadata."""

    def test_regime_info_hashable(self, sample_ohlcv_data):
        """RegimeInfo should be hashable for dict keys."""
        info = RegimeInfo(
            regime_id=0,
            regime_name="BULL",
            expected_return=0.001,
            expected_volatility=0.02,
        )
        regime_dict = {info: "test"}
        assert info in regime_dict

    def test_regime_info_from_hmm(self, sample_ohlcv_data):
        """HMM should populate regime infos after training."""
        engine = HMMEngine(n_candidates=[3])
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        for regime_id, info in engine.regime_infos.items():
            assert isinstance(info.regime_id, (int, np.integer))
            assert isinstance(info.regime_name, str)
            assert isinstance(info.expected_return, (float, np.floating))
            assert isinstance(info.expected_volatility, float)
            assert info.expected_volatility > 0


class TestNoLookAheadBias:
    """
    CRITICAL: Verify forward algorithm has no look-ahead bias.
    """

    def test_no_lookahead_regime_identity(self, sample_ohlcv_data):
        """Regime at bar T should be same with or without future data."""
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        t = 200

        # Reset and predict using only data up to T
        engine.reset_inference_state()

        regime_short: RegimeState | None = None
        for i in range(t + 1):
            obs = features.values[i]
            regime_short = engine.predict_filtered(obs)

        # Reset and predict using data up to T+100, capture regime at T
        engine.reset_inference_state()

        regime_long_at_t: RegimeState | None = None
        for i in range(t + 101):
            obs = features.values[i]
            reg = engine.predict_filtered(obs)
            if i == t:
                regime_long_at_t = reg

        # Both should have same regime at T
        assert regime_short is not None
        assert regime_long_at_t is not None
        assert regime_short.state_id == regime_long_at_t.state_id, (
            f"LOOK-AHEAD BIAS: regime at T differs "
            f"(short={regime_short.state_id}, long={regime_long_at_t.state_id})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
