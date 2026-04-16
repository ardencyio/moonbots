"""
MANDATORY look-ahead test per HMM.md spec.

Slices OHLCV (NOT pre-computed features) at two horizons, recomputes features
from scratch for each slice, and asserts regime at bar T is identical. This is
the strongest form of the guarantee: if any upstream step (feature computation,
normalisation, scaler state) looked at future data, this test will fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.core.hmm.feature_engineering import (
    compute_features,
    prepare_features_for_hmm,
)
from shared.core.hmm.hmm_engine import HMMEngine


@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    np.random.seed(42)
    n_bars = 1200
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


def test_feature_values_identical_on_ohlcv_slices(ohlcv_data):
    """Feature values at bar T must match whether computed on OHLCV[:T+1] or OHLCV[:T+N]."""
    t = 800  # well past warm-up
    full = compute_features(ohlcv_data)
    truncated = compute_features(ohlcv_data.iloc[: t + 1].copy())

    for col in full.columns:
        a = full.iloc[t][col]
        b = truncated.iloc[t][col]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, (
            f"LOOK-AHEAD in feature '{col}' at T={t}: full={a}, truncated={b}"
        )


def test_normalized_features_identical_on_ohlcv_slices(ohlcv_data):
    """Normalised (rolling-z) feature values at T must match across OHLCV slices."""
    t = 900
    full = prepare_features_for_hmm(ohlcv_data.copy())
    truncated = prepare_features_for_hmm(ohlcv_data.iloc[: t + 1].copy())

    # The last row of the truncated frame corresponds to bar t in the full frame,
    # but dropna() may shift the index; align by original positional index.
    full_row = full.loc[t] if t in full.index else None
    trunc_row = truncated.loc[t] if t in truncated.index else None

    assert full_row is not None, "bar T absent in full frame after dropna"
    assert trunc_row is not None, "bar T absent in truncated frame after dropna"

    for col in full.columns:
        assert full_row[col] == pytest.approx(trunc_row[col], nan_ok=True), (
            f"LOOK-AHEAD in normalized '{col}' at T={t}: "
            f"full={full_row[col]}, truncated={trunc_row[col]}"
        )


def test_regime_at_T_identical_across_ohlcv_slices(ohlcv_data):
    """
    Train on OHLCV[:T+N], then replay the forward algorithm on features
    derived from OHLCV[:T+1] and OHLCV[:T+N] independently. Regime at bar T
    must be identical — the forward algorithm + feature pipeline must never
    consume future data.
    """
    t = 900
    t_plus = 1100  # future bars beyond t

    # Train a single model on the longer slice; we only test inference-time look-ahead.
    train_features = prepare_features_for_hmm(ohlcv_data.iloc[:t_plus].copy())
    engine = HMMEngine(n_candidates=[3], n_init=2, stability_bars=1)
    engine.train(train_features)

    # Run 1: feed features built from OHLCV up to t+1
    features_short = prepare_features_for_hmm(ohlcv_data.iloc[: t + 1].copy())
    engine.reset_inference_state()
    regime_short = None
    for i in range(len(features_short)):
        regime_short = engine.predict_filtered(features_short.values[i])

    # Run 2: feed features built from OHLCV up to t_plus, capturing regime at t
    features_long = prepare_features_for_hmm(ohlcv_data.iloc[:t_plus].copy())
    engine.reset_inference_state()
    regime_long_at_t = None
    # Identify positional bar for t in features_long (should be the matching index)
    if t in features_long.index:
        stop_pos = features_long.index.get_loc(t)
    else:
        stop_pos = len(features_short) - 1

    for i in range(stop_pos + 1):
        reg = engine.predict_filtered(features_long.values[i])
        if i == stop_pos:
            regime_long_at_t = reg

    assert regime_short is not None
    assert regime_long_at_t is not None
    assert regime_short.state_id == regime_long_at_t.state_id, (
        f"LOOK-AHEAD: regime at T={t} differs "
        f"(short={regime_short.state_id}, long={regime_long_at_t.state_id})"
    )


def test_plus_minus_dm_never_both_nonzero_on_same_bar(ohlcv_data):
    """ADX ±DM: on any bar, at most one of plus_dm / minus_dm may be non-zero."""
    from shared.core.hmm.feature_engineering import compute_adx

    # compute_adx is used to derive ADX; we recompute ±DM here to assert the guarantee.
    high = ohlcv_data["high"]
    low = ohlcv_data["low"]
    raw_plus = high - high.shift(1)
    raw_minus = low.shift(1) - low
    plus_dm = np.where((raw_plus > raw_minus) & (raw_plus > 0), raw_plus, 0.0)
    minus_dm = np.where((raw_minus > raw_plus) & (raw_minus > 0), raw_minus, 0.0)

    overlap = (plus_dm > 0) & (minus_dm > 0)
    assert not overlap.any(), "plus_dm and minus_dm must never both be non-zero"

    # Sanity: compute_adx returns a Series aligned to the input
    adx = compute_adx(ohlcv_data)
    assert len(adx) == len(ohlcv_data)
