"""
Feature Engineering for HMM Regime Detection.

Computes technical features from OHLCV data for HMM regime classification.
All features are computed as pure functions with NO look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all HMM features from OHLCV data.

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

    Returns:
        DataFrame of standardized features for HMM input
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    df["return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["return_20"] = np.log(df["close"] / df["close"].shift(20))

    df["realized_vol"] = df["return_1"].rolling(20).std()
    df["volatility_ratio"] = (
        df["return_1"].rolling(5).std() / df["return_1"].rolling(20).std()
    )

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["price_vs_sma20"] = df["close"] / df["sma_20"]
    df["price_vs_sma50"] = df["close"] / df["sma_50"]
    df["sma20_vs_sma50"] = df["sma_20"] / df["sma_50"]
    df["dist_from_sma200"] = (df["close"] - df["sma_200"]) / df["close"]
    df["sma50_slope"] = df["sma_50"].pct_change(5)

    df["roc_10"] = df["close"].pct_change(10)
    df["roc_20"] = df["close"].pct_change(20)

    # Raw RSI only — normalize_features() applies the rolling z-score downstream,
    # so emitting a pre-z-scored column would standardise the signal twice.
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr"] / df["close"]

    df["volume_ma50"] = df["volume"].rolling(50).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_zscore"] = (df["volume"] - df["volume_ma50"]) / df["volume"].rolling(
        50
    ).std()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    df["volume_trend"] = df["volume_ma20"].pct_change(10)

    df["adx"] = compute_adx(df)

    df["up_down"] = (df["close"] > df["close"].shift(1)).astype(int)
    df["up_days_ratio"] = df["up_down"].rolling(20).mean()

    # Do NOT use ffill()/bfill() — they would fabricate warm-up data. Warm-up NaN
    # rows are dropped by prepare_features_for_hmm() downstream.

    hmm_features = [
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

    return df[hmm_features]


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Compute both +/-DM masks from the RAW series before writing either back.
    # Reading a mutated plus_dm in the minus_dm branch would weaken the
    # condition and let both be non-zero on the same bar, violating ADX.
    raw_plus = high - high.shift(1)
    raw_minus = low.shift(1) - low

    plus_dm_arr = np.where((raw_plus > raw_minus) & (raw_plus > 0), raw_plus, 0.0)
    minus_dm_arr = np.where((raw_minus > raw_plus) & (raw_minus > 0), raw_minus, 0.0)

    plus_dm = pd.Series(plus_dm_arr, index=high.index)
    minus_dm = pd.Series(minus_dm_arr, index=high.index)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, 1)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, 1)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
    adx = dx.rolling(period).mean()

    return adx


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features using CAUSAL rolling z-score to handle non-stationarity.

    CRITICAL: Uses 252-period rolling mean/std with shift(1) to prevent look-ahead bias.
    Global StandardScaler is PROHIBITED - it uses future data in the fit.

    Args:
        features: DataFrame of raw features

    Returns:
        DataFrame of standardized features with NaN in warm-up period
    """
    window = 252  # Trading year
    min_periods = 60  # Minimum data points for rolling calc

    # Causal rolling z-score: use shifted mean/std from prior bars only
    # shift(1) ensures we don't use current bar's own stats
    rolling_mean = features.rolling(window, min_periods=min_periods).mean().shift(1)
    rolling_std = features.rolling(window, min_periods=min_periods).std().shift(1)

    normalized = (features - rolling_mean) / rolling_std

    return normalized


def prepare_features_for_hmm(
    df: pd.DataFrame,
    use_normalization: bool = True,
) -> pd.DataFrame:
    """
    Full feature preparation pipeline for HMM.

    Args:
        df: OHLCV DataFrame
        use_normalization: Whether to normalize features

    Returns:
        Ready-to-use features DataFrame
    """
    features = compute_features(df)

    if use_normalization:
        features = normalize_features(features)

    # Drop any remaining NaN/Inf
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()

    return features
