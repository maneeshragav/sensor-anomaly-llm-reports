"""
LLM-based incident report generation, via a locally-running Ollama server.

This module talks to Ollama's real HTTP API (POST /api/generate). It is
written to run against a real local Ollama install -- it does NOT bundle or
fake a model. If Ollama isn't reachable (e.g. it isn't running, or you're
executing this in an environment with no access to your machine's
localhost), `generate_incident_report` raises a clear OllamaUnavailable
error, and the CLI falls back to a clearly-labeled template report instead
of silently pretending an LLM wrote it.

Usage (once Ollama is installed and running locally):
    ollama pull llama3.2          # one-time, pulls the model
    ollama serve                  # usually already running as a background service
    python src/main.py            # will call this module automatically
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaUnavailable(RuntimeError):
    pass


@dataclass
class AnomalyEvent:
    start: str
    end: str
    peak_value: float
    baseline_mean: float
    baseline_std: float
    detector: str


PROMPT_TEMPLATE = """You are an industrial reliability engineer writing a short incident report \
for a machine-temperature sensor anomaly detected by an automated monitoring system.

Anomaly details:
- Detected by: {detector}
- Time window: {start} to {end}
- Peak sensor reading during the window: {peak_value:.2f}
- Normal baseline mean/std before the anomaly: {baseline_mean:.2f} / {baseline_std:.2f}

Write a concise incident report (4-6 sentences) for a maintenance team that:
1. States plainly what was detected and when.
2. Explains, in plain language, how far the reading deviated from normal.
3. Suggests one or two concrete next steps for a technician to check.

Do not invent a root cause you cannot know from the numbers above -- describe \
the anomaly and recommend investigation, rather than diagnosing a specific fault.
"""


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 60) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailable(
            "Could not reach Ollama at "
            f"{OLLAMA_BASE_URL}. Make sure Ollama is installed and running "
            f"locally (`ollama serve`) and that the model is pulled "
            f"(`ollama pull {model}`)."
        ) from exc

    if resp.status_code != 200:
        raise OllamaUnavailable(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if "response" not in data:
        raise OllamaUnavailable(f"Unexpected Ollama response shape: {json.dumps(data)[:300]}")

    return data["response"].strip()


def generate_incident_report(event: AnomalyEvent, model: str = DEFAULT_MODEL) -> str:
    """Generate a natural-language incident report for one detected anomaly event."""
    prompt = PROMPT_TEMPLATE.format(
        detector=event.detector,
        start=event.start,
        end=event.end,
        peak_value=event.peak_value,
        baseline_mean=event.baseline_mean,
        baseline_std=event.baseline_std,
    )
    return _call_ollama(prompt, model=model)


def template_fallback_report(event: AnomalyEvent) -> str:
    """
    Deterministic, non-LLM report used only when Ollama is unavailable, so the
    pipeline still produces something usable end-to-end. Clearly labeled as a
    fallback wherever it's written out -- never presented as LLM output.
    """
    deviation = (
        (event.peak_value - event.baseline_mean) / event.baseline_std
        if event.baseline_std
        else 0.0
    )
    return (
        f"[TEMPLATE FALLBACK - Ollama unavailable] Anomaly detected by {event.detector} "
        f"between {event.start} and {event.end}. Peak reading {event.peak_value:.2f} vs. "
        f"baseline {event.baseline_mean:.2f} (+/-{event.baseline_std:.2f}), a deviation of "
        f"about {deviation:.1f} standard deviations from normal. Recommend a technician "
        f"inspect the sensor and surrounding equipment for this time window."
    )
