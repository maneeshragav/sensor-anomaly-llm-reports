"""
Self-contained tests for the pipeline. Run with:
    python3 tests/test_pipeline.py

These cover:
  - feature engineering doesn't blow up / produces expected columns
  - both anomaly detectors fire on a synthetic injected spike
  - evaluate.py correctly scores a detector against known labels
  - llm_report.py correctly talks to Ollama's real API contract (verified
    against a mock server standing in for Ollama, since the real Ollama
    instance runs on the user's own machine and isn't reachable from here)
  - llm_report.py fails informatively when no server is listening at all
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from features import build_features, FEATURE_COLUMNS
from detect import run_detection
from evaluate import evaluate_detector, extract_events
from llm_report import generate_incident_report, template_fallback_report, AnomalyEvent, OllamaUnavailable
from mock_ollama_server import start_mock_server


def make_synthetic_series() -> pd.DataFrame:
    """Flat baseline with one obvious injected spike, so detectors have
    something unambiguous to catch and we can check they actually catch it."""
    rng = np.random.default_rng(0)
    n = 500
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
    values = 75 + rng.normal(0, 0.3, size=n)  # normal operating noise
    values[250:255] = 95  # injected anomaly: a clear 5-sample spike
    return pd.DataFrame({"timestamp": timestamps, "value": values})


def test_features():
    df = make_synthetic_series()
    feat = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in feat.columns, f"missing feature column {col}"
    assert feat["value"].isna().sum() == 0
    print("PASS: test_features")


def test_detection_catches_injected_spike():
    df = make_synthetic_series()
    feat = build_features(df)
    result = run_detection(feat)
    spike_window = result.iloc[248:258]
    assert spike_window["is_anomaly"].any(), "neither detector flagged the injected spike"
    print("PASS: test_detection_catches_injected_spike")


def test_evaluate_scores_correctly():
    df = make_synthetic_series()
    feat = build_features(df)
    result = run_detection(feat)
    label_ts = [str(df["timestamp"].iloc[252])]  # the middle of the injected spike
    ev = evaluate_detector(result, "is_anomaly", label_ts, detector_name="consensus")
    assert ev.true_positives == 1, f"expected 1 true positive, got {ev.true_positives}"
    assert ev.false_negatives == 0
    print(f"PASS: test_evaluate_scores_correctly ({ev})")


def test_extract_events():
    df = make_synthetic_series()
    feat = build_features(df)
    result = run_detection(feat)
    events = extract_events(result, "is_anomaly")
    assert len(events) >= 1
    assert events[0]["peak_value"] > events[0]["baseline_mean"]
    print(f"PASS: test_extract_events ({len(events)} event(s) found)")


def test_llm_report_against_mock_server():
    server = start_mock_server(port=11434)
    time.sleep(0.2)
    try:
        event = AnomalyEvent(
            start="2024-01-01 20:50:00",
            end="2024-01-01 21:10:00",
            peak_value=95.0,
            baseline_mean=75.0,
            baseline_std=0.3,
            detector="iforest",
        )
        report = generate_incident_report(event)
        assert "MOCK RESPONSE" in report, f"unexpected report content: {report}"
        print(f"PASS: test_llm_report_against_mock_server -> {report}")
    finally:
        server.shutdown()


def test_llm_report_unavailable_is_informative():
    event = AnomalyEvent(
        start="2024-01-01 20:50:00", end="2024-01-01 21:10:00",
        peak_value=95.0, baseline_mean=75.0, baseline_std=0.3, detector="iforest",
    )
    try:
        generate_incident_report(event)
        raise AssertionError("expected OllamaUnavailable when nothing is listening")
    except OllamaUnavailable as e:
        assert "ollama serve" in str(e)
        print(f"PASS: test_llm_report_unavailable_is_informative -> {e}")

    fallback = template_fallback_report(event)
    assert "TEMPLATE FALLBACK" in fallback
    print(f"PASS: template_fallback_report -> {fallback}")


if __name__ == "__main__":
    test_features()
    test_detection_catches_injected_spike()
    test_evaluate_scores_correctly()
    test_extract_events()
    # Order matters: unavailable-server test first (nothing listening yet),
    # then the mock-server test starts its own server on the same port.
    test_llm_report_unavailable_is_informative()
    test_llm_report_against_mock_server()
    print("\nALL TESTS PASSED")
