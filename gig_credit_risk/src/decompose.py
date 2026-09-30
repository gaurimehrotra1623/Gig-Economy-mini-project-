"""
decompose.py

Core of the research idea: for each worker's income series, run STL
decomposition and extract features that separate "predictable" volatility
(trend + seasonal) from "true risk" volatility (residual).

income_t = trend_t + seasonal_t + residual_t

Baseline features (what a naive model would use):
    - mean_income
    - raw_std          (std dev of raw income -- treats ALL volatility as risk)
    - raw_cv           (coefficient of variation = raw_std / mean_income)

Decomposed features (your hypothesis: these should be better risk predictors):
    - trend_slope           (is income secularly rising or falling?)
    - seasonal_amplitude    (how big is the predictable seasonal swing?)
    - residual_std          (std dev of the LEFTOVER noise -- the "true risk" signal)
    - residual_cv           (residual_std / mean_income, normalized)
    - max_abs_residual      (worst single unexplained shock in the series)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def extract_features_for_worker(income_series: pd.Series, period: int = 52) -> dict:
    """
    Run STL decomposition on one worker's weekly income series and return
    a dict of baseline + decomposed features.

    period=52 assumes weekly data with an annual seasonal cycle. If your
    real data is shorter than ~2 seasonal periods, STL will be unreliable --
    you need at least ~104 weeks (2 years) for a period=52 decomposition to
    be meaningful. For shorter histories, consider period=4 (monthly-ish
    cycles within a quarter) or switch to a simpler moving-average baseline.
    """
    income_series = income_series.reset_index(drop=True)
    mean_income = income_series.mean()
    raw_std = income_series.std()

    features = {
        "mean_income": mean_income,
        "raw_std": raw_std,
        "raw_cv": raw_std / mean_income if mean_income > 0 else np.nan,
    }

    if len(income_series) < 2 * period:
        # Not enough history for a reliable seasonal decomposition
        features.update({
            "trend_slope": np.nan,
            "seasonal_amplitude": np.nan,
            "residual_std": np.nan,
            "residual_cv": np.nan,
            "max_abs_residual": np.nan,
        })
        return features

    stl = STL(income_series, period=period, robust=True)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    # Trend slope: simple linear fit to the trend component
    x = np.arange(len(trend))
    trend_slope = np.polyfit(x, trend, 1)[0]

    features.update({
        "trend_slope": trend_slope,
        "seasonal_amplitude": seasonal.max() - seasonal.min(),
        "residual_std": resid.std(),
        "residual_cv": resid.std() / mean_income if mean_income > 0 else np.nan,
        "max_abs_residual": resid.abs().max(),
    })
    return features


def build_feature_table(income_df: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    """
    income_df: long-format DataFrame with columns ['worker_id', 'week', 'income']
    Returns: one row per worker with baseline + decomposed features.
    """
    rows = []
    for worker_id, group in income_df.groupby("worker_id"):
        group = group.sort_values("week")
        feats = extract_features_for_worker(group["income"], period=period)
        feats["worker_id"] = worker_id
        # carry through archetype label if present (useful for validation, drop before modeling)
        if "archetype" in group.columns:
            feats["archetype"] = group["archetype"].iloc[0]
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    income_df = pd.read_csv("data/simulated_income.csv")
    feature_df = build_feature_table(income_df, period=52)
    feature_df.to_csv("data/worker_features.csv", index=False)

    print(feature_df.shape)
    print(feature_df.groupby("archetype")[
        ["raw_cv", "residual_cv", "seasonal_amplitude"]
    ].mean())
