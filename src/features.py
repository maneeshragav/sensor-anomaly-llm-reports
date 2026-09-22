"""
Feature engineering for the sensor time-series.

Turns the raw (timestamp, value) readings into a set of rolling statistical
features that make anomalies (sudden shifts, drift, spikes) easier for a
model to separate from normal operating noise.
"""
from __future__ import annotations

import pandas as pd


def load_series(csv_path: str) -> pd.DataFrame:
    """Load the raw sensor CSV into a clean, time-indexed DataFrame."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame, short_window: int = 6, long_window: int = 576) -> pd.DataFrame:
    """
    Build rolling-window features on top of the raw sensor value.

    short_window / long_window are expressed in number of samples. The NAB
    machine-temperature series is sampled every 5 minutes, so a short_window
    of 6 ~= 30 minutes and a long_window of 576 ~= 2 days -- long enough to
    smooth over daily cycles, which a shorter window (tried during tuning,
    see tests/tune.py) mistook for anomalies far too often.
    """
    out = df.copy()

    out["roll_mean_short"] = out["value"].rolling(short_window, min_periods=1).mean()
    out["roll_std_short"] = out["value"].rolling(short_window, min_periods=1).std().fillna(0)

    out["roll_mean_long"] = out["value"].rolling(long_window, min_periods=1).mean()
    out["roll_std_long"] = out["value"].rolling(long_window, min_periods=1).std().fillna(0)

    # Rate of change between consecutive readings.
    out["diff"] = out["value"].diff().fillna(0)

    # How far the current reading sits from its recent baseline, in units of
    # that baseline's own volatility (a self-normalizing "surprise" score).
    out["z_short"] = (out["value"] - out["roll_mean_short"]) / out["roll_std_short"].replace(0, 1e-6)
    out["z_long"] = (out["value"] - out["roll_mean_long"]) / out["roll_std_long"].replace(0, 1e-6)

    return out


FEATURE_COLUMNS = [
    "value",
    "roll_mean_short",
    "roll_std_short",
    "roll_mean_long",
    "roll_std_long",
    "diff",
    "z_short",
    "z_long",
]
