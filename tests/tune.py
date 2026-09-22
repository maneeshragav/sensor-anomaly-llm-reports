import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from features import load_series, build_features
from detect import zscore_baseline, isolation_forest_detect
from evaluate import evaluate_detector

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = load_series(os.path.join(ROOT, "data", "machine_temperature_system_failure.csv"))
with open(os.path.join(ROOT, "data", "combined_labels.json")) as f:
    labels = json.load(f)["realKnownCause/machine_temperature_system_failure.csv"]

merge_gap = pd.Timedelta(hours=6)

best = []
for long_window in [36, 144, 288, 576]:
    feat = build_features(df, short_window=6, long_window=long_window)
    for z_thresh in [3.0, 3.5, 4.0, 5.0]:
        flags = zscore_baseline(feat, threshold=z_thresh)
        tmp = feat.copy(); tmp["flag"] = flags
        ev = evaluate_detector(tmp, "flag", labels, merge_gap=merge_gap, detector_name=f"zscore(lw={long_window},th={z_thresh})")
        best.append((ev.f1, ev.precision, ev.recall, ev.detector, ev.n_events))
    for contamination in [0.0005, 0.001, 0.002, 0.005]:
        flag, score = isolation_forest_detect(feat, contamination=contamination)
        tmp = feat.copy(); tmp["flag"] = flag
        ev = evaluate_detector(tmp, "flag", labels, merge_gap=merge_gap, detector_name=f"iforest(lw={long_window},c={contamination})")
        best.append((ev.f1, ev.precision, ev.recall, ev.detector, ev.n_events))

best.sort(reverse=True)
for f1, p, r, name, n in best[:15]:
    print(f"f1={f1:.3f} precision={p:.3f} recall={r:.3f} events={n:3d}  {name}")
