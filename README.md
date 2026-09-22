# Sensor Anomaly Detection & LLM Incident Reporting

A predictive-maintenance-style pipeline that detects anomalies in a real industrial sensor stream and uses a locally-run LLM (via [Ollama](https://ollama.com)) to turn each detected anomaly into a plain-language incident report a maintenance team could actually read.

## What this does

1. **Ingests real sensor data** — the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) `machine_temperature_system_failure` dataset: ~22,700 temperature readings from an industrial machine over ~2.5 months, with 4 human-labeled real failure events.
2. **Engineers features** — short/long rolling means, rolling volatility, rate of change, and self-normalizing z-scores against both a short and a long (2-day) baseline.
3. **Detects anomalies** with two independent approaches, so there's something to compare rather than a single black-box number:
   - A statistical z-score baseline.
   - An **Isolation Forest** trained on the engineered features.
4. **Evaluates both detectors** against the real labeled failures with a transparent, tolerance-window precision/recall/F1 metric (see [Evaluation methodology](#evaluation-methodology) — this is *not* the official NAB scoring algorithm, which is more elaborate).
5. **Generates a natural-language incident report per detected event** by calling a locally-running Ollama model, with prompt engineering that constrains the model to describe the anomaly and recommend investigation rather than inventing a root cause. If Ollama isn't reachable, it falls back to a clearly-labeled deterministic template so the pipeline still runs end-to-end.
6. **Visualizes everything** in a Streamlit dashboard — the raw sensor trace, detected events, ground-truth labels, evaluation metrics, and the generated reports.

## Results

On the real, held-out ground-truth labels (4 known failure events):

| Detector | Precision | Recall | F1 | Events flagged |
|---|---|---|---|---|
| Z-score baseline | 0.038 | 0.25 | 0.067 | 26 |
| **Isolation Forest** | **0.50** | **0.50** | **0.50** | 4 |

The Isolation Forest model, trained on engineered rolling-window features, catches half of the real labeled failures while raising only 2 false alarms across 2.5 months of data — a large, honest improvement over the naive z-score baseline, which drowns in false positives. These numbers come straight out of `reports/evaluation_report.json` after running the pipeline; nothing here is hand-picked or rounded up.

Tuning process (`tests/tune.py`) swept the rolling-window length and detection thresholds for both detectors; the long rolling window mattered far more than the exact threshold — a 2-day window smooths over the series' daily cycle, which a short window kept mistaking for anomalies.

## Architecture

```
src/
  features.py    -- rolling-window feature engineering
  detect.py      -- z-score baseline + Isolation Forest detectors
  evaluate.py     -- tolerance-window precision/recall/F1 against labels
  llm_report.py  -- Ollama client + prompt template + template fallback
  main.py        -- orchestrates the full pipeline, writes reports/
  dashboard.py   -- Streamlit UI
tests/
  test_pipeline.py     -- unit tests (features, detection, evaluation, LLM client)
  mock_ollama_server.py -- stands in for Ollama's API in tests
  tune.py               -- parameter sweep used to pick the defaults above
data/
  machine_temperature_system_failure.csv  -- real NAB sensor data
  combined_labels.json                    -- NAB's human-labeled anomaly timestamps
reports/            -- generated: evaluation_report.json, incident_reports.md, detection_output.csv
```

## Setup

```bash
pip install -r requirements.txt
```

To get real (not template-fallback) LLM-generated reports, you need [Ollama](https://ollama.com) installed and running **locally**, since the report generator calls `http://localhost:11434`:

```bash
ollama pull llama3.2
ollama serve      # usually already running as a background service after install
```

## Run

```bash
python3 src/main.py          # runs detection + evaluation + report generation, writes reports/
streamlit run src/dashboard.py   # interactive dashboard
```

## Tests

```bash
python3 tests/test_pipeline.py
```

Covers feature engineering, both detectors against a synthetic injected spike, the evaluation scoring logic, and the Ollama client's request/response handling — the last of these against `mock_ollama_server.py`, a tiny stand-in that mimics Ollama's real `/api/generate` response shape, since a live Ollama instance running on your own machine can't be reached from every environment this test suite might run in (e.g. CI).

## Evaluation methodology

NAB's official scoring algorithm is an application-weighted windowed score. This project uses a simpler, fully transparent alternative instead, implemented in `evaluate.py`:

1. Consecutive (or near-consecutive, within a merge gap) flagged timestamps are grouped into detected "events".
2. An event is a true positive if it falls within a tolerance window of any labeled failure timestamp; each label can only be claimed once.
3. Unclaimed labels are false negatives; unmatched events are false positives.

This is a design choice made for transparency, not a claim of matching NAB's published benchmark numbers.

## Honest limitations

- The z-score baseline is intentionally simple; it is not tuned beyond the sweep in `tests/tune.py`, since its purpose here is to be a naive comparison point, not a competitive detector.
- Recall of 0.5 means real failures are still being missed — a maintenance team using this as-is would want a second signal (e.g. vibration or acoustic sensors) rather than relying on temperature alone.
- The LLM report-generation prompt explicitly tells the model not to invent a root cause, but as with any LLM output, a human should review a generated report before acting on it.
