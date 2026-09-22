"""
End-to-end pipeline: load data -> engineer features -> detect anomalies ->
evaluate against ground truth -> generate an incident report per detected
event (via Ollama if reachable, otherwise a clearly-labeled template
fallback) -> save everything under reports/.

Run:
    python3 src/main.py
"""
from __future__ import annotations

import json
import os

import pandas as pd

from features import load_series, build_features
from detect import run_detection
from evaluate import evaluate_detector, extract_events
from llm_report import generate_incident_report, template_fallback_report, AnomalyEvent, OllamaUnavailable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "machine_temperature_system_failure.csv")
LABELS_JSON = os.path.join(ROOT, "data", "combined_labels.json")
REPORTS_DIR = os.path.join(ROOT, "reports")
LABEL_KEY = "realKnownCause/machine_temperature_system_failure.csv"


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading {DATA_CSV} ...")
    df = load_series(DATA_CSV)
    print(f"  {len(df)} readings, {df['timestamp'].min()} -> {df['timestamp'].max()}")

    with open(LABELS_JSON) as f:
        labels_all = json.load(f)
    ground_truth = labels_all[LABEL_KEY]
    print(f"  {len(ground_truth)} ground-truth labeled anomalies: {ground_truth}")

    print("\nBuilding features ...")
    feat = build_features(df)

    print("Running detectors (z-score baseline + Isolation Forest) ...")
    result = run_detection(feat)
    print(f"  z-score flags: {int(result['zscore_flag'].sum())} points")
    print(f"  iforest flags: {int(result['iforest_flag'].sum())} points")

    print("\nEvaluating against ground truth (tolerance = 4 hours, merge_gap = 6 hours) ...")
    merge_gap = pd.Timedelta(hours=6)
    metrics = {}
    for col, name in [("zscore_flag", "zscore_baseline"), ("iforest_flag", "isolation_forest_primary")]:
        ev = evaluate_detector(result, col, ground_truth, detector_name=name, merge_gap=merge_gap)
        metrics[name] = ev.__dict__
        print(f"  {name:18s} precision={ev.precision:.3f} recall={ev.recall:.3f} f1={ev.f1:.3f} "
              f"(TP={ev.true_positives} FP={ev.false_positives} FN={ev.false_negatives}, {ev.n_events} events)")

    with open(os.path.join(REPORTS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved evaluation_report.json")

    print("\nExtracting Isolation Forest anomaly events and generating incident reports ...")
    events = extract_events(result, "is_anomaly", merge_gap=merge_gap)
    print(f"  {len(events)} events detected")

    report_lines = ["# Incident Reports\n"]
    ollama_used = False
    for i, ev in enumerate(events, 1):
        event = AnomalyEvent(
            start=ev["start"], end=ev["end"], peak_value=ev["peak_value"],
            baseline_mean=ev["baseline_mean"], baseline_std=ev["baseline_std"],
            detector="Isolation Forest",
        )
        try:
            text = generate_incident_report(event)
            ollama_used = True
            source = "Ollama LLM"
        except OllamaUnavailable as e:
            text = template_fallback_report(event)
            source = "template fallback (Ollama unreachable)"
        report_lines.append(f"## Event {i}: {ev['start']} to {ev['end']}\n")
        report_lines.append(f"*Source: {source}*\n")
        report_lines.append(text + "\n")

    with open(os.path.join(REPORTS_DIR, "incident_reports.md"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"Saved incident_reports.md ({'used live Ollama' if ollama_used else 'used template fallback -- Ollama not reachable from this environment'})")

    result.to_csv(os.path.join(REPORTS_DIR, "detection_output.csv"), index=False)
    print("Saved detection_output.csv")


if __name__ == "__main__":
    main()
