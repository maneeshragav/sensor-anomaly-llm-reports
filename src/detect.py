"""
Anomaly detection algorithms.

Two independent detectors are implemented deliberately, so the project has
something to compare and evaluate rather than a single black-box score:

1. A statistical z-score baseline (fast, interpretable, no training).
2. An Isolation Forest model (learns the shape of "normal" from the
   engineered features and isolates points that are structurally unusual).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import FEATURE_COLUMNS


def zscore_baseline(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Flag points whose long-window z-score exceeds `threshold` in magnitude."""
    return (df["z_long"].abs() > threshold)


def isolation_forest_detect(
    df: pd.DataFrame,
    contamination: float = 0.002,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Fit an Isolation Forest on the engineered features and return:
      - a boolean Series of flagged anomalies
      - a continuous anomaly score (higher = more anomalous) for ranking/plots
    """
    X = df[FEATURE_COLUMNS].to_numpy()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    raw_scores = model.decision_function(X)  # higher = more normal
    anomaly_score = -raw_scores  # flip so higher = more anomalous
    is_anomaly = model.predict(X) == -1

    return pd.Series(is_anomaly, index=df.index), pd.Series(anomaly_score, index=df.index)


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Run both detectors and attach their outputs as columns."""
    out = df.copy()
    out["zscore_flag"] = zscore_baseline(out)
    if_flag, if_score = isolation_forest_detect(out)
    out["iforest_flag"] = if_flag
    out["iforest_score"] = if_score
    # Isolation Forest is the primary detector -- tuning (tests/tune.py)
    # showed the z-score baseline is far noisier on this series (best
    # f1=0.087 vs. Isolation Forest's 0.5), so it's kept only as a reported
    # comparison point, not folded into the "official" flag used downstream.
    out["is_anomaly"] = out["iforest_flag"]
    return out
