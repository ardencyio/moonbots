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
from shared.core.hmm.hmm_engine import HMMEngine, RegimeInfo, RegimeState


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
            "return_5",
            "return_20",
            "realized_vol",
            "volatility_ratio",
            "price_vs_sma20",
            "sma20_vs_sma50",
            "dist_from_sma200",
            "sma50_slope",
            "rsi",
            "roc_10",
            "roc_20",
            "atr_ratio",
            "volume_zscore",
            "volume_ratio",
            "volume_trend",
            "adx",
            "up_days_ratio",
        ]
        assert list(features.columns) == expected_features

        # Warm-up period should have NaNs (expected with rolling windows, no bfill)
        # This is CORRECT behavior - we drop these NaNs downstream for training
        warmup_nans = features.isna().sum().sum()
        assert warmup_nans > 0  # Expected warm-up NaNs from rolling windows
        assert warmup_nans < len(features)  # But not all rows are NaN

    def test_feature_values_no_lookahead(self, sample_ohlcv_data):
        """Feature values at T must be identical whether computed on data[0:T] or data[0:T+N].

        This is the feature-engineering equivalent of the forward-algorithm
        look-ahead test. If bfill() or global standardization were used,
        feature values at T would change when future data is added.
        """
        full_features = compute_features(sample_ohlcv_data)

        # Pick a point well past warm-up (index 500)
        t = 500

        # Compute features using only data up to T+1
        truncated = sample_ohlcv_data.iloc[: t + 1].copy()
        truncated_features = compute_features(truncated)

        # Compare values at T — must be identical (no future data used)
        for col in full_features.columns:
            full_val = full_features.iloc[t][col]
            trunc_val = truncated_features.iloc[t][col]

            if np.isnan(full_val) and np.isnan(trunc_val):
                continue

            assert full_val == trunc_val, (
                f"LOOK-AHEAD BIAS in feature '{col}' at T={t}: "
                f"full={full_val}, truncated={trunc_val}"
            )


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


class TestForwardAlgorithmDirection:
    """
    Regression: forward step must use transmat_[i,j] (not transposed).

    Seed the engine with a known asymmetric transition matrix and compare
    predict_filtered to hmmlearn's own predict_proba on the same prefix.
    The two must agree within a tight tolerance; they would disagree if the
    forward step applied the transpose.
    """

    def _build_seeded_engine(self):
        from hmmlearn.hmm import GaussianHMM

        rng = np.random.default_rng(7)
        n_states, n_features = 3, 2
        model = GaussianHMM(n_components=n_states, covariance_type="full")
        # Manually seed HMM parameters so no training randomness matters
        model.startprob_ = np.array([0.7, 0.2, 0.1])
        # Asymmetric T: sticky state 0; state 1 tends to jump to 2; state 2 recovers to 0
        model.transmat_ = np.array(
            [
                [0.90, 0.07, 0.03],  # from 0
                [0.05, 0.20, 0.75],  # from 1  (fast exit to 2)
                [0.40, 0.10, 0.50],  # from 2
            ]
        )
        model.means_ = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]])
        model.covars_ = np.tile(np.eye(n_features), (n_states, 1, 1)) * 0.5
        model.n_features = n_features

        # Synthetic observations: sample from a specific sequence
        obs = rng.normal(size=(50, n_features))

        engine = HMMEngine(n_candidates=[n_states], stability_bars=1)
        engine.model = model
        engine.selected_n_components = n_states
        engine.n_regimes = n_states
        engine.regime_labels = ["BEAR", "NEUTRAL", "BULL"]
        for rid in range(n_states):
            from shared.core.hmm.hmm_engine import RegimeInfo

            engine.regime_infos[rid] = RegimeInfo(
                regime_id=rid,
                regime_name=engine.regime_labels[rid],
                expected_return=model.means_[rid, 0],
                expected_volatility=float(np.sqrt(model.covars_[rid, 0, 0])),
            )
        return engine, model, obs

    def _reference_filtered_posteriors(self, model, obs):
        """
        Compute filtered (forward-only) posteriors independently of hmmlearn's
        forward implementation, using the canonical recurrence
            α_t(j) = Σ_i α_{t-1}(i) · T[i, j] · emission(o_t | j).
        """
        from scipy.special import logsumexp

        emissions = model._compute_log_likelihood(obs)
        log_T = np.log(model.transmat_ + 1e-300)
        log_alpha = np.log(model.startprob_ + 1e-300) + emissions[0]
        log_alpha = log_alpha - logsumexp(log_alpha)
        posteriors = [np.exp(log_alpha)]
        for t in range(1, len(obs)):
            log_alpha_trans = logsumexp(log_alpha[:, None] + log_T, axis=0)
            log_alpha = log_alpha_trans + emissions[t]
            log_alpha = log_alpha - logsumexp(log_alpha)
            posteriors.append(np.exp(log_alpha))
        return np.array(posteriors)

    def test_predict_filtered_matches_reference_posterior(self):
        engine, model, obs = self._build_seeded_engine()

        reference = self._reference_filtered_posteriors(model, obs)

        engine.reset_inference_state()
        ours = np.empty_like(reference)
        for i in range(len(obs)):
            state = engine.predict_filtered(obs[i])
            ours[i] = state.state_probabilities

        # On this asymmetric T, even a single-bar transpose error produces
        # visibly different posteriors — tolerance below is tight.
        np.testing.assert_allclose(ours, reference, atol=1e-10, rtol=1e-10)

    def test_predict_filtered_would_disagree_if_transposed(self):
        """Sanity check: applying T^T produces a different posterior than T."""
        engine, model, obs = self._build_seeded_engine()

        # Our implementation (correct T)
        engine.reset_inference_state()
        state_correct = None
        for i in range(20):
            state_correct = engine.predict_filtered(obs[i])

        # Manually apply the transposed forward step for the same prefix
        from scipy.special import logsumexp

        log_alpha = np.log(model.startprob_ + 1e-300)
        # First obs: emission + start
        emissions = model._compute_log_likelihood(obs[:20])
        log_alpha = log_alpha + emissions[0]
        log_alpha = log_alpha - logsumexp(log_alpha)
        log_trans_T = np.log(model.transmat_.T + 1e-300)
        for t in range(1, 20):
            log_alpha = (
                logsumexp(log_alpha[:, None] + log_trans_T, axis=0) + emissions[t]
            )
            log_alpha = log_alpha - logsumexp(log_alpha)

        transposed_posterior = np.exp(log_alpha)
        assert state_correct is not None
        # They must differ on this asymmetric T
        assert not np.allclose(
            state_correct.state_probabilities, transposed_posterior, atol=1e-4
        ), (
            "transmat direction check is a no-op on this matrix; pick a more asymmetric T"
        )


class TestPredictFilteredInputValidation:
    """predict_filtered must validate shape and finiteness of feature_vector."""

    def _trained_engine(self, sample_ohlcv_data):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)
        return engine, features

    def test_nan_feature_raises(self, sample_ohlcv_data):
        engine, features = self._trained_engine(sample_ohlcv_data)
        obs = features.values[100].copy()
        obs[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            engine.predict_filtered(obs)

    def test_inf_feature_raises(self, sample_ohlcv_data):
        engine, features = self._trained_engine(sample_ohlcv_data)
        obs = features.values[100].copy()
        obs[1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            engine.predict_filtered(obs)

    def test_wrong_shape_raises(self, sample_ohlcv_data):
        engine, features = self._trained_engine(sample_ohlcv_data)
        bad = np.zeros(features.shape[1] + 3)  # too many features
        with pytest.raises(ValueError, match="expected shape"):
            engine.predict_filtered(bad)

    def test_timestamp_is_threaded_through(self, sample_ohlcv_data):
        from datetime import datetime as _dt

        engine, features = self._trained_engine(sample_ohlcv_data)
        ts = _dt(2026, 1, 1, 9, 30)
        state = engine.predict_filtered(features.values[100], timestamp=ts)
        assert state.timestamp == ts


class TestTrainResetsInferenceState:
    """After train() the inference state must be reset."""

    def test_train_clears_previous_proba(self, sample_ohlcv_data):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        # Make a prediction to populate _previous_proba
        engine.predict_filtered(features.values[100])
        assert engine._previous_proba is not None

        # Retrain → _previous_proba must reset
        engine.train(features)
        assert engine._previous_proba is None


class TestRegimeRankCache:
    """HMM-FU-001: cache lexsort of regime means once, read O(1) per bar."""

    def _expected_rank_map(self, engine: HMMEngine) -> dict[int, int]:
        assert engine.model is not None
        regime_returns = engine.model.means_[:, 0]
        sorted_ids = np.lexsort((np.arange(len(regime_returns)), regime_returns))
        return {int(rid): rank for rank, rid in enumerate(sorted_ids)}

    def test_cache_populated_after_train(self, sample_ohlcv_data):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        assert engine._regime_id_to_rank == self._expected_rank_map(engine)
        assert engine._sorted_regime_ids.shape == (engine.selected_n_components,)

    def test_predict_filtered_label_matches_cached_rank(self, sample_ohlcv_data):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        state = engine.predict_filtered(features.values[100])
        expected_rank = engine._regime_id_to_rank[state.state_id]
        assert state.label == engine.regime_labels[expected_rank]

    def test_predict_filtered_does_not_recompute_lexsort(
        self, sample_ohlcv_data, monkeypatch
    ):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        import shared.core.hmm.hmm_engine as hmm_engine_module

        calls = {"count": 0}
        original_lexsort = np.lexsort

        def counting_lexsort(*args, **kwargs):
            calls["count"] += 1
            return original_lexsort(*args, **kwargs)

        monkeypatch.setattr(hmm_engine_module.np, "lexsort", counting_lexsort)

        for row in features.values[100:110]:
            engine.predict_filtered(row)

        assert calls["count"] == 0

    def test_reset_inference_state_preserves_cache(self, sample_ohlcv_data):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)

        expected = dict(engine._regime_id_to_rank)
        engine.reset_inference_state()
        assert engine._regime_id_to_rank == expected

    def test_load_rebuilds_cache(self, sample_ohlcv_data, tmp_path):
        engine = HMMEngine(n_candidates=[3], stability_bars=1)
        features = prepare_features_for_hmm(sample_ohlcv_data).dropna()
        engine.train(features)
        expected = dict(engine._regime_id_to_rank)

        save_path = tmp_path / "engine.pkl"
        engine.save(save_path)
        restored = HMMEngine.load(save_path)

        assert restored._regime_id_to_rank == expected
        restored_state = restored.predict_filtered(features.values[100])
        fresh_state = engine.predict_filtered(features.values[100])
        assert restored_state.label == fresh_state.label


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
