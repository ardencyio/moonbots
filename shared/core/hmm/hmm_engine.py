"""
HMM Regime Detection Engine.

This implements a Gaussian HMM with automatic model selection via BIC.
CRITICAL: Uses FORWARD ALGORITHM only (filtered inference) to prevent look-ahead bias.
The Viterbi algorithm (model.predict()) revises past states using future data - this is look-ahead bias.
Instead, we compute P(state_t | observations_1:t) using only past and present data.

Design Philosophy:
- The HMM is a VOLATILITY CLASSIFIER, not a price direction predictor
- Regimes are labeled by mean return for human readability
- The STRATEGY layer sorts by VOLATILITY independently for allocation decisions
- Labels do NOT drive strategy decisions
"""

from __future__ import annotations

import math
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


@dataclass
class RegimeInfo:
    """Information about a regime from the trained HMM."""

    regime_id: int
    regime_name: str
    expected_return: float  # Mean return in this regime
    expected_volatility: float  # Std dev of returns
    probability: float = 1.0  # Posterior probability (updated at inference time)
    recommended_strategy_type: Optional[str] = None
    max_leverage_allowed: float = 1.0
    max_position_size_pct: float = 1.0
    min_confidence_to_act: float = 0.55

    def __hash__(self):
        return hash(self.regime_id)


@dataclass
class RegimeState:
    """Current regime state at inference time."""

    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray  # Full distribution over states
    timestamp: datetime
    is_confirmed: bool = False  # True after stability_bars consecutive detections
    consecutive_bars: int = 0  # Consecutive bars in this regime
    min_confidence: float = 0.55  # Minimum probability threshold for certainty

    @property
    def is_uncertain(self) -> bool:
        """Return True if regime probability is below confidence threshold."""
        return self.probability < self.min_confidence


class HMMEngine:
    """
    HMM Regime Detection Engine with Forward Algorithm (No Look-Ahead).

    Attributes:
        n_candidates: Number of regime candidates to test (e.g., [3,4,5,6,7])
        n_init: Number of random initializations per candidate
        covariance_type: HMM covariance type ("full", "tied", "diag", "spherical")
        min_train_bars: Minimum bars required for training
        stability_bars: Bars needed to confirm regime change
        flicker_window: Window size for flicker detection
        flicker_threshold: Max regime changes per window before uncertainty mode
        min_confidence: Minimum probability confidence threshold
    """

    def __init__(
        self,
        n_candidates: Optional[list[int]] = None,
        n_init: int = 10,
        covariance_type: str = "full",
        min_train_bars: int = 252,
        stability_bars: int = 3,
        flicker_window: int = 20,
        flicker_threshold: int = 4,
        min_confidence: float = 0.55,
    ):
        self.n_candidates = n_candidates or [3, 4, 5, 6, 7]
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_train_bars = min_train_bars
        self.stability_bars = stability_bars
        self.flicker_window = flicker_window
        self.flicker_threshold = flicker_threshold
        self.min_confidence = min_confidence

        # Trained model attributes
        self.model: Optional[GaussianHMM] = None
        self.n_regimes: int = 0  # Alias for selected_n_components
        self.selected_n_components: int = 0
        self.best_bic: float = float("inf")  # Best BIC score
        self.bic_scores: dict[int, float] = {}
        self.regime_infos: dict[int, RegimeInfo] = {}
        self.regime_labels: list[str] = []

        # Inference state
        self._current_regime: Optional[int] = None
        self._consecutive_bars: int = 0
        self._previous_proba: Optional[np.ndarray] = None
        self._regime_history: list[int] = []
        self._is_flickering: bool = False

        # Metadata
        self._training_date: Optional[datetime] = None
        self._training_data_shape: tuple = (0, 0)

    def train(self, features: pd.DataFrame) -> dict:
        """
        Train HMM with automatic model selection via BIC.

        Args:
            features: DataFrame of features (rows=time, cols=features)

        Returns:
            Training metadata dict with BIC scores, selected model, etc.

        Raises:
            ValueError: If insufficient data for training
        """
        if len(features) < self.min_train_bars * 2:
            raise ValueError(
                f"Insufficient data: {len(features)} bars, "
                f"minimum {self.min_train_bars * 2} required"
            )

        feature_matrix = features.values
        logger.info(
            f"Training HMM with {len(feature_matrix)} bars, "
            f"{feature_matrix.shape[1]} features"
        )

        best_bic = float("inf")
        best_model = None
        best_n_components = 0

        self.bic_scores = {}

        for n_components in self.n_candidates:
            logger.info(f"Testing n_components={n_components}")

            # Create HMM model
            hmm = GaussianHMM(
                n_components=n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_init,
                random_state=42,
            )

            try:
                hmm.fit(feature_matrix)

                # Compute BIC: -2*log_likelihood + n_params*log(n_samples)
                log_likelihood = hmm.score(feature_matrix)
                n_params = self._count_parameters(hmm, feature_matrix.shape[1])
                n_samples = feature_matrix.shape[0]
                bic = -2 * log_likelihood + n_params * math.log(n_samples)

                self.bic_scores[n_components] = bic
                logger.info(f"  BIC: {bic:.2f} (likelihood: {log_likelihood:.2f})")

                if bic < best_bic:
                    best_bic = bic
                    best_model = hmm
                    best_n_components = n_components

            except Exception as e:
                logger.warning(f"  FAILED n_components={n_components}: {e}")
                continue

        if best_model is None:
            raise RuntimeError("HMM training failed for all candidate models")

        self.model = best_model
        self.selected_n_components = best_n_components
        self.n_regimes = best_n_components
        self.best_bic = best_bic
        self._training_date = datetime.now()
        self._training_data_shape = feature_matrix.shape

        logger.info(
            f"Selected model: {best_n_components} regimes (BIC: {best_bic:.2f})"
        )

        # Label regimes and create regime infos
        self._label_regimes()

        return {
            "selected_n_components": best_n_components,
            "bic_score": best_bic,
            "bic_scores": self.bic_scores,
            "training_date": self._training_date.isoformat(),
            "training_bars": len(feature_matrix),
        }

    def _count_parameters(self, model: GaussianHMM, n_features: int) -> int:
        """Count free parameters in HMM for BIC calculation."""
        n = model.n_components
        k = n_features

        # Transition matrix: n*(n-1) free parameters (rows sum to 1)
        trans_params = n * (n - 1)

        # Start probabilities: n-1 free parameters
        start_params = n - 1

        # Emission parameters depend on covariance type
        if model.covariance_type == "full":
            # Mean: n*k, Covariance: n * k*(k+1)/2 (symmetric)
            emit_params = n * k + n * k * (k + 1) // 2
        elif model.covariance_type == "tied":
            # Shared covariance: k*(k+1)/2, per-component means: n*k
            emit_params = k * (k + 1) // 2 + n * k
        elif model.covariance_type == "diag":
            # Diagonal covariance: n*k (means) + n*k (variances)
            emit_params = 2 * n * k
        elif model.covariance_type == "spherical":
            # Spherical covariance: n*k (means) + n (variances)
            emit_params = n * (k + 1)
        else:
            raise ValueError(f"Unknown covariance type: {model.covariance_type}")

        return trans_params + start_params + emit_params

    def _label_regimes(self) -> None:
        """
        Label regimes by expected return (ascending) for human readability.

        Note: Labels are for display only. Strategy layer sorts by VOLATILITY independently.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        n = self.selected_n_components

        # Get mean returns per regime
        regime_returns = self.model.means_[
            :, 0
        ]  # First feature is typically log return

        # Sort regime IDs by return (ascending)
        sorted_ids = np.argsort(regime_returns)

        # Generate labels based on regime count
        if n == 3:
            labels = ["BEAR", "NEUTRAL", "BULL"]
        elif n == 4:
            labels = ["CRASH", "BEAR", "BULL", "EUPHORIA"]
        elif n == 5:
            labels = ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"]
        elif n == 6:
            labels = [
                "CRASH",
                "STRONG_BEAR",
                "WEAK_BEAR",
                "WEAK_BULL",
                "STRONG_BULL",
                "EUPHORIA",
            ]
        elif n == 7:
            labels = [
                "CRASH",
                "STRONG_BEAR",
                "WEAK_BEAR",
                "NEUTRAL",
                "WEAK_BULL",
                "STRONG_BULL",
                "EUPHORIA",
            ]
        else:
            labels = [f"REGIME_{i}" for i in range(n)]

        # Map sorted IDs to labels (lowest return -> first label)
        self.regime_labels = labels

        # Create RegimeInfo for each regime
        for rank, regime_id in enumerate(sorted_ids):
            regime_id = int(regime_id)  # Cast np.int64 to plain int
            regime_name = labels[rank]
            expected_return = regime_returns[regime_id]

            # Compute expected volatility (std of returns in this regime)
            # Use diagonal of covariance matrix (variance of first feature = return)
            if self.model.covariance_type == "full":
                variance = self.model.covars_[regime_id, 0, 0]
            elif self.model.covariance_type in ("diag", "spherical"):
                variance = self.model.covars_[regime_id, 0]
            else:
                variance = np.mean(self.model.covars_[:, 0, 0])
            expected_volatility = np.sqrt(max(0, variance))

            self.regime_infos[regime_id] = RegimeInfo(
                regime_id=regime_id,
                regime_name=regime_name,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
            )

        logger.info(f"Regimes labeled: {self.regime_labels}")
        for rid, info in self.regime_infos.items():
            logger.info(
                f"  {info.regime_name} (id={rid}): "
                f"return={info.expected_return * 100:.3f}%, "
                f"vol={info.expected_volatility * 100:.3f}%"
            )

    def predict_filtered(
        self,
        feature_vector: np.ndarray,
    ) -> RegimeState:
        """
        Predict regime using FORWARD ALGORITHM only (filtered inference).

        CRITICAL: This uses ONLY past and present data. NO FUTURE DATA.
        This prevents look-ahead bias that plagues model.predict() (Viterbi).

        Computes: P(state_t | observations_1:t) via forward recursion:
            alpha_0 = startprob * emission(obs_0)
            alpha_t = (alpha_{t-1} @ transmat) * emission(obs_t)

        Args:
            feature_vector: Single observation vector (1D array of features)

        Returns:
            RegimeState with current regime label, probability, confirmation status
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        obs = np.atleast_2d(feature_vector)  # Shape: (1, n_features)
        model = self.model

        # --- FORWARD ALGORITHM ---
        # Work in log space for numerical stability

        # Step 1: Compute emission log-probabilities for current observation
        # For GaussianHMM: log N(obs | mean, cov)
        log_emissions = np.empty(model.n_components)

        for i in range(model.n_components):
            # Compute Mahalanobis distance: (obs-mu)^T @ cov^{-1} @ (obs-mu)
            diff = obs - model.means_[i]

            if model.covariance_type == "full":
                cov = model.covars_[i]
                # Use scipy linalg for stable inverse
                from scipy.linalg import solve

                try:
                    # Solve cov @ x = diff.T, then mahalanobis = diff @ x
                    cov_inv_diff = solve(cov, diff.T, assume_a="pos")
                    mahalanobis = np.sum(diff @ cov_inv_diff)
                    log_det = np.log(np.linalg.det(cov))
                except np.linalg.LinAlgError:
                    log_emissions[i] = -np.inf
                    continue
            elif model.covariance_type == "tied":
                cov = model.covars_[0]
                from scipy.linalg import solve

                cov_inv_diff = solve(cov, diff.T, assume_a="pos")
                mahalanobis = np.sum(diff @ cov_inv_diff)
                log_det = np.log(np.linalg.det(cov))
            elif model.covariance_type == "diag":
                var = model.covars_[i]
                mahalanobis = np.sum((diff**2) / var)
                log_det = np.sum(np.log(var))
            elif model.covariance_type == "spherical":
                var = model.covars_[i]
                mahalanobis = np.sum(diff**2) / var
                log_det = model.means_.shape[1] * np.log(var)
            else:
                raise ValueError(f"Unknown covariance type: {model.covariance_type}")

            n_features = model.means_.shape[1]
            log_emit = -0.5 * (mahalanobis + log_det + n_features * np.log(2 * np.pi))
            log_emissions[i] = log_emit

        # Step 2: Forward recursion
        # alpha[i] = log P(state=i | observations_1:t)
        # For first obs: alpha_0 = startprob * emission
        # For t > 0: alpha_t[j] = emission_j * sum_i (alpha_{t-1}[i] * trans[i,j])

        if self._previous_proba is not None:
            # Use cached previous posterior as input
            log_alpha_prev = np.log(self._previous_proba + 1e-300)

            # Apply transition: logsumexp over previous states
            log_transposed = np.log(
                model.transmat_.T + 1e-300
            )  # Shape: (n_states, n_states)
            # Result shape: (n_states,) - probability of being in each state after transition
            log_alpha_transitioned = logsumexp(
                log_alpha_prev[:, np.newaxis] + log_transposed, axis=0
            )
        else:
            # First observation: use start distribution directly
            log_alpha_transitioned = np.log(model.startprob_ + 1e-300)

        # Add emission probabilities
        log_alpha_unnorm = log_alpha_transitioned + log_emissions

        # Normalize (log-space)
        log_normalizer = logsumexp(log_alpha_unnorm)
        log_alpha = log_alpha_unnorm - log_normalizer

        # Step 3: Compute posterior distribution
        posterior = np.exp(log_alpha)

        # Store for next iteration
        self._previous_proba = posterior

        # Step 4: Determine current regime (max probability)
        current_regime = int(np.argmax(posterior))
        regime_prob = float(posterior[current_regime])

        # Step 5: Check regime stability
        regime_changed = current_regime != self._current_regime

        if regime_changed:
            self._consecutive_bars = 1
            self._current_regime = current_regime
        else:
            self._consecutive_bars += 1

        # Step 6: Update regime history for flicker detection
        self._regime_history.append(current_regime)
        if len(self._regime_history) > self.flicker_window:
            self._regime_history.pop(0)

        # Check flicker rate
        if len(self._regime_history) >= self.flicker_window:
            # Count regime changes in window
            changes = sum(
                1
                for i in range(1, len(self._regime_history))
                if self._regime_history[i] != self._regime_history[i - 1]
            )
            self._is_flickering = changes > self.flicker_threshold

        # Confirmation status
        is_confirmed = self._consecutive_bars >= self.stability_bars

        # Get regime label
        # Map regime_id to label via sorted order
        if self.model:
            sorted_ids = np.argsort(self.model.means_[:, 0])
            rank = np.where(sorted_ids == current_regime)[0][0]
            label = (
                self.regime_labels[rank]
                if rank < len(self.regime_labels)
                else f"UNKNOWN_{current_regime}"
            )
        else:
            label = f"REGIME_{current_regime}"

        logger.debug(
            f"Regime: {label} (id={current_regime}), prob={regime_prob:.3f}, "
            f"consecutive={self._consecutive_bars}, confirmed={is_confirmed}"
        )

        return RegimeState(
            label=label,
            state_id=current_regime,
            probability=regime_prob,
            state_probabilities=posterior,
            timestamp=datetime.now(),
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
            min_confidence=self.min_confidence,
        )

    def get_regime_flicker_rate(self) -> float:
        """Get regime changes per flicker_window."""
        if len(self._regime_history) < 2:
            return 0.0
        changes = sum(
            1
            for i in range(1, len(self._regime_history))
            if self._regime_history[i] != self._regime_history[i - 1]
        )
        return changes

    def is_flickering(self) -> bool:
        """Check if regime is flickering (too many changes in window)."""
        return self._is_flickering

    def get_transition_matrix(self) -> np.ndarray:
        """Get learned transition probability matrix."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.transmat_.copy()

    def reset_inference_state(self) -> None:
        """Reset inference state for new data stream."""
        self._previous_proba = None
        self._current_regime = None
        self._consecutive_bars = 0
        self._regime_history.clear()
        self._is_flickering = False

    def save(self, path: str | Path) -> None:
        """Save trained model and metadata to file."""
        if self.model is None:
            raise RuntimeError("No trained model to save")

        data = {
            "model": self.model,
            "selected_n_components": self.selected_n_components,
            "bic_scores": self.bic_scores,
            "regime_infos": {
                k: {
                    attr: getattr(regime, attr)
                    for attr in [
                        "regime_id",
                        "regime_name",
                        "expected_return",
                        "expected_volatility",
                    ]
                }
                for k, regime in self.regime_infos.items()
            },
            "regime_labels": self.regime_labels,
            "training_date": self._training_date.isoformat()
            if self._training_date
            else None,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "HMMEngine":
        """Load trained model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        engine = HMMEngine()
        engine.model = data["model"]
        engine.selected_n_components = data["selected_n_components"]
        engine.bic_scores = data["bic_scores"]
        engine.regime_labels = data["regime_labels"]
        engine._training_date = (
            datetime.fromisoformat(data["training_date"])
            if data.get("training_date")
            else None
        )

        # Reconstruct regime_infos
        for info_dict in data["regime_infos"].values():
            regime = RegimeInfo(**info_dict)
            engine.regime_infos[regime.regime_id] = regime

        return engine
