"""
Evaluation against NAB's ground-truth anomaly labels.

NAB ships a small set of *point* timestamps per file marking when a human
labeler judged an anomaly to have occurred (see labels/combined_labels.json
in the NAB repo). The official NAB scoring algorithm is a more elaborate,
application-profile-weighted windowed score; this project implements a
simpler, fully transparent tolerance-window evaluation instead, so the
metrics below are NOT the official "NAB score" -- they're a straightforward
precision/recall/F1 computed as follows:

1. Consecutive flagged timestamps are grouped into detected "events".
2. An event counts as a true positive if it falls within `tolerance` of any
   labeled timestamp; each label can only be "claimed" once.
3. Any label with no matching event is a false negative.
4. Any event that doesn't match a label is a false positive.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class EvalResult:
    detector: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    matched_labels: list
    unmatched_labels: list
    n_events: int


def _group_events(
    flags: pd.Series,
    timestamps: pd.Series,
    merge_gap: pd.Timedelta = pd.Timedelta(0),
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Collapse a boolean flag series into (start, end) timestamp spans of
    contiguous flags. `merge_gap` optionally merges two flagged spans
    separated by a quiet gap no longer than `merge_gap` into a single event,
    so one drawn-out incident that flickers on/off isn't counted as dozens
    of separate events.
    """
    raw_events = []
    in_event = False
    start = None
    prev_ts = None
    for ts, flag in zip(timestamps, flags):
        if flag and not in_event:
            in_event = True
            start = ts
        elif not flag and in_event:
            in_event = False
            raw_events.append((start, prev_ts))
        prev_ts = ts
    if in_event:
        raw_events.append((start, prev_ts))

    if not raw_events or merge_gap <= pd.Timedelta(0):
        return raw_events

    merged = [raw_events[0]]
    for start, end in raw_events[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= merge_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def extract_events(df: pd.DataFrame, flag_col: str, lookback: int = 36, merge_gap: pd.Timedelta = pd.Timedelta(0)) -> list[dict]:
    """
    Turn a boolean flag column into a list of event dicts with the summary
    stats needed to write an incident report: the time window, the peak
    value during the event, and the "normal" baseline mean/std computed from
    the `lookback` samples immediately before the event started.
    """
    events = _group_events(df[flag_col], df["timestamp"], merge_gap=merge_gap)
    out = []
    for start, end in events:
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        window = df.loc[mask]

        start_idx = df.index[df["timestamp"] == start][0]
        baseline = df.loc[max(0, start_idx - lookback):start_idx - 1, "value"]
        if baseline.empty:
            baseline = df.loc[:start_idx, "value"]

        out.append(
            {
                "start": str(start),
                "end": str(end),
                "peak_value": float(window["value"].abs().max()),
                "baseline_mean": float(baseline.mean()),
                "baseline_std": float(baseline.std() or 0.0),
            }
        )
    return out


def evaluate_detector(
    df: pd.DataFrame,
    flag_col: str,
    label_timestamps: list[str],
    tolerance: pd.Timedelta = pd.Timedelta(hours=4),
    detector_name: str = "detector",
    merge_gap: pd.Timedelta = pd.Timedelta(0),
) -> EvalResult:
    labels = [pd.Timestamp(t) for t in label_timestamps]
    events = _group_events(df[flag_col], df["timestamp"], merge_gap=merge_gap)

    matched_labels: set = set()
    tp = 0
    fp = 0

    for start, end in events:
        window_start, window_end = start - tolerance, end + tolerance
        hit = None
        for label in labels:
            if label in matched_labels:
                continue
            if window_start <= label <= window_end:
                hit = label
                break
        if hit is not None:
            matched_labels.add(hit)
            tp += 1
        else:
            fp += 1

    fn = len(labels) - len(matched_labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvalResult(
        detector=detector_name,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        matched_labels=[str(t) for t in sorted(matched_labels)],
        unmatched_labels=[str(t) for t in labels if t not in matched_labels],
        n_events=len(events),
    )
