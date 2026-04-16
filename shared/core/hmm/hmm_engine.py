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
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import LinAlgError as ScipyLinAlgError
from scipy.linalg import solve
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
        min_train_bars: Half the minimum total bars required for training (actual minimum = min_train_bars * 2, default 504)
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

        # Thread-local inference state (for multi-threaded safety)
        self._thread_local = threading.local()

        # Metadata
        self._training_date: Optional[datetime] = None
        self._training_data_shape: tuple = (0, 0)

    def _get_inference_state(self):
        """Get thread-local inference state, initializing if needed."""
        state = getattr(self._thread_local, "state", None)
        if state is None:
            state = {
                "current_regime": None,
                "consecutive_bars": 0,
                "previous_proba": None,
                "regime_history": deque(maxlen=self.flicker_window),
                "is_flickering": False,
            }
            self._thread_local.state = state
        return state

    @property
    def _current_regime(self):
        return self._get_inference_state()["current_regime"]

    @_current_regime.setter
    def _current_regime(self, value):
        self._get_inference_state()["current_regime"] = value

    @property
    def _consecutive_bars(self):
        return self._get_inference_state()["consecutive_bars"]

    @_consecutive_bars.setter
    def _consecutive_bars(self, value):
        self._get_inference_state()["consecutive_bars"] = value

    @property
    def _previous_proba(self):
        return self._get_inference_state()["previous_proba"]

    @_previous_proba.setter
    def _previous_proba(self, value):
        self._get_inference_state()["previous_proba"] = value

    @property
    def _regime_history(self):
        return self._get_inference_state()["regime_history"]

    @property
    def _is_flickering(self):
        return self._get_inference_state()["is_flickering"]

    @_is_flickering.setter
    def _is_flickering(self, value):
        self._get_inference_state()["is_flickering"] = value

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

        self.bic_scores = {}
        # (bic, n_components, model, converged) per candidate; we prefer converged
        converged_candidates: list[tuple[float, int, GaussianHMM]] = []
        non_converged_candidates: list[tuple[float, int, GaussianHMM]] = []

        for n_components in self.n_candidates:
            logger.info(f"Testing n_components={n_components}")

            # Run n_init random restarts and keep the best by log-likelihood
            candidate_bic = float("inf")
            candidate_model: Optional[GaussianHMM] = None
            candidate_converged = False

            for init_idx in range(self.n_init):
                hmm = GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    n_iter=200,
                    random_state=42 + init_idx,
                )

                try:
                    hmm.fit(feature_matrix)
                    log_likelihood = hmm.score(feature_matrix)
                    n_params = self._count_parameters(hmm, feature_matrix.shape[1])
                    n_samples = feature_matrix.shape[0]
                    bic = -2 * log_likelihood + n_params * math.log(n_samples)

                    converged = bool(getattr(hmm.monitor_, "converged", False))
                    # Prefer converged fits; break ties by BIC.
                    if (converged and not candidate_converged) or (
                        converged == candidate_converged and bic < candidate_bic
                    ):
                        candidate_bic = bic
                        candidate_model = hmm
                        candidate_converged = converged

                except Exception as e:
                    logger.debug(
                        f"  n_components={n_components} init={init_idx} FAILED: {e}"
                    )
                    continue

            if candidate_model is not None:
                self.bic_scores[n_components] = candidate_bic
                logger.info(
                    "  BIC=%.2f, converged=%s (best of %d inits)",
                    candidate_bic,
                    candidate_converged,
                    self.n_init,
                )
                bucket = (
                    converged_candidates
                    if candidate_converged
                    else non_converged_candidates
                )
                bucket.append((candidate_bic, n_components, candidate_model))
            else:
                logger.warning(
                    f"  All {self.n_init} inits FAILED for n_components={n_components}"
                )

        pool = converged_candidates or non_converged_candidates
        if not pool:
            raise RuntimeError("HMM training failed for all candidate models")
        if not converged_candidates:
            logger.warning(
                "No HMM candidate converged; using best non-converged model as fallback"
            )

        best_bic, best_n_components, best_model = min(pool, key=lambda row: row[0])

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

        # Fresh model invalidates any prior inference state (may differ in n_components)
        self.reset_inference_state()
        logger.info("Inference state reset after train()")

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

        # Sort regime IDs by return (ascending) with stable secondary key (regime_id)
        # Use lexsort for deterministic ordering when returns are tied
        # lexsort sorts by last key first, so (regime_id, regime_returns) sorts by returns, breaks ties by id
        sorted_ids = np.lexsort((np.arange(len(regime_returns)), regime_returns))

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
        timestamp: Optional[datetime] = None,
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
            timestamp: Bar timestamp (defaults to datetime.now() if omitted).
                Backtests MUST pass the bar time to avoid baking wall clock
                into backtest state.

        Returns:
            RegimeState with current regime label, probability, confirmation status

        Raises:
            RuntimeError: Model not trained.
            ValueError: feature_vector has wrong shape or non-finite entries.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        model = self.model
        obs = np.atleast_2d(np.asarray(feature_vector, dtype=np.float64))
        expected_shape = (1, model.n_features)
        if obs.shape != expected_shape:
            raise ValueError(
                f"predict_filtered expected shape {expected_shape}; got {obs.shape}"
            )
        if not np.all(np.isfinite(obs)):
            raise ValueError(
                "predict_filtered received non-finite values in feature_vector"
            )
        if (
            self._previous_proba is not None
            and len(self._previous_proba) != model.n_components
        ):
            logger.warning(
                "Stale inference state detected (prev proba size=%d, model n_components=%d); resetting",
                len(self._previous_proba),
                model.n_components,
            )
            self.reset_inference_state()

        # --- FORWARD ALGORITHM ---
        # Work in log space for numerical stability

        # Step 1: Compute emission log-probabilities for current observation
        # For GaussianHMM: log N(obs | mean, cov)
        log_emissions = np.empty(model.n_components)

        n_features = model.n_features
        for i in range(model.n_components):
            # Compute Mahalanobis distance: (obs-mu)^T @ cov^{-1} @ (obs-mu)
            diff = obs - model.means_[i]

            if model.covariance_type == "full":
                cov = model.covars_[i]
                try:
                    cov_inv_diff = solve(cov, diff.T, assume_a="pos")
                    mahalanobis = float(np.sum(diff @ cov_inv_diff))
                    sign, logdet = np.linalg.slogdet(cov)
                    if sign <= 0 or not np.isfinite(logdet):
                        log_emissions[i] = -np.inf
                        continue
                    log_det = logdet
                except (np.linalg.LinAlgError, ScipyLinAlgError, ValueError):
                    log_emissions[i] = -np.inf
                    continue
            elif model.covariance_type == "tied":
                cov = model.covars_[0]
                try:
                    cov_inv_diff = solve(cov, diff.T, assume_a="pos")
                    mahalanobis = float(np.sum(diff @ cov_inv_diff))
                    sign, logdet = np.linalg.slogdet(cov)
                    if sign <= 0 or not np.isfinite(logdet):
                        log_emissions[i] = -np.inf
                        continue
                    log_det = logdet
                except (np.linalg.LinAlgError, ScipyLinAlgError, ValueError):
                    log_emissions[i] = -np.inf
                    continue
            elif model.covariance_type == "diag":
                var = model.covars_[i]
                mahalanobis = float(np.sum((diff**2) / var))
                log_det = float(np.sum(np.log(var)))
            elif model.covariance_type == "spherical":
                var = model.covars_[i]
                mahalanobis = float(np.sum(diff**2) / var)
                log_det = n_features * float(np.log(var))
            else:
                raise ValueError(f"Unknown covariance type: {model.covariance_type}")

            log_emissions[i] = -0.5 * (
                mahalanobis + log_det + n_features * np.log(2 * np.pi)
            )

        # Step 2: Forward recursion
        # alpha[i] = log P(state=i | observations_1:t)
        # For first obs: alpha_0 = startprob * emission
        # For t > 0: alpha_t[j] = emission_j * sum_i (alpha_{t-1}[i] * trans[i,j])

        if self._previous_proba is not None:
            # Use cached previous posterior as input
            log_alpha_prev = np.log(self._previous_proba + 1e-300)

            # Forward transition step:
            # P(state_t=j) = Σ_i P(state_{t-1}=i) * P(state_t=j | state_{t-1}=i)
            #
            # hmmlearn stores transmat_[i, j] = P(state_t=j | state_{t-1}=i).
            # Do NOT transpose: element [i, j] of (log_alpha_prev[:, None] + log_trans)
            # is log α_{t-1}[i] + log T[i, j]. Summing over axis=0 sums over the
            # previous state i, yielding log α_t[j].
            log_trans = np.log(model.transmat_ + 1e-300)
            log_alpha_transitioned = logsumexp(
                log_alpha_prev[:, np.newaxis] + log_trans, axis=0
            )
        else:
            # First observation: use start distribution directly
            log_alpha_transitioned = np.log(model.startprob_ + 1e-300)

        # Add emission probabilities
        log_alpha_unnorm = log_alpha_transitioned + log_emissions

        # Normalize (log-space)
        log_normalizer = logsumexp(log_alpha_unnorm)

        # Numerical stability: guard against overflow/underflow
        if not np.isfinite(log_normalizer):
            logger.warning(
                "Numerical underflow in forward step; rebooting to uniform prior"
            )
            n_states = model.n_components
            log_alpha_unnorm = np.full(n_states, -np.log(n_states))
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

        # Step 6: Update regime history for flicker detection (deque auto-trims)
        self._regime_history.append(current_regime)

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

        # Get regime label using same lexsort as _label_regimes for consistency
        if self.model:
            regime_returns = self.model.means_[:, 0]
            sorted_ids = np.lexsort((np.arange(len(regime_returns)), regime_returns))
            rank = int(np.where(sorted_ids == current_regime)[0][0])
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
            timestamp=timestamp if timestamp is not None else datetime.now(),
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
        """Reset inference state for new data stream (thread-local)."""
        state = self._get_inference_state()
        state["previous_proba"] = None
        state["current_regime"] = None
        state["consecutive_bars"] = 0
        state["regime_history"] = deque(maxlen=self.flicker_window)
        state["is_flickering"] = False

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
                    "regime_id": regime.regime_id,
                    "regime_name": regime.regime_name,
                    "expected_return": regime.expected_return,
                    "expected_volatility": regime.expected_volatility,
                    "probability": regime.probability,
                    "recommended_strategy_type": regime.recommended_strategy_type,
                    "max_leverage_allowed": regime.max_leverage_allowed,
                    "max_position_size_pct": regime.max_position_size_pct,
                    "min_confidence_to_act": regime.min_confidence_to_act,
                }
                for k, regime in self.regime_infos.items()
            },
            "regime_labels": self.regime_labels,
            "training_date": self._training_date.isoformat()
            if self._training_date
            else None,
            # Hyperparameters for consistent reload
            "n_candidates": self.n_candidates,
            "n_init": self.n_init,
            "covariance_type": self.covariance_type,
            "min_train_bars": self.min_train_bars,
            "stability_bars": self.stability_bars,
            "flicker_window": self.flicker_window,
            "flicker_threshold": self.flicker_threshold,
            "min_confidence": self.min_confidence,
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

        # Restore hyperparameters from saved data
        engine.n_candidates = data.get("n_candidates", [3, 4, 5, 6, 7])
        engine.n_init = data.get("n_init", 10)
        engine.covariance_type = data.get("covariance_type", "full")
        engine.min_train_bars = data.get("min_train_bars", 252)
        engine.stability_bars = data.get("stability_bars", 3)
        engine.flicker_window = data.get("flicker_window", 20)
        engine.flicker_threshold = data.get("flicker_threshold", 4)
        engine.min_confidence = data.get("min_confidence", 0.55)

        # Restore derived attributes
        engine.n_regimes = engine.selected_n_components
        engine.best_bic = (
            min(engine.bic_scores.values()) if engine.bic_scores else float("inf")
        )

        # Reconstruct regime_infos
        for info_dict in data["regime_infos"].values():
            regime = RegimeInfo(**info_dict)
            engine.regime_infos[regime.regime_id] = regime

        return engine
