"""Deliberately weak controls from the SentinelAI case (before/after contrast)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinelai.config import SENSORS
from sentinelai.data.windows import WindowBatch, make_windows


@dataclass
class WeakDecision:
    action: str
    score: float
    auto_tripped: bool


def forward_fill_only(df: pd.DataFrame, sensors: list[str] | None = None) -> pd.DataFrame:
    """Anti-pattern: forward-fill gaps, no mask or staleness signal."""
    sensors = sensors or SENSORS
    out = df.copy()
    out[sensors] = out[sensors].ffill().bfill()
    return out


def weak_score_from_window(values: np.ndarray) -> float:
    """
    Naive hazard proxy: mean absolute z-score across sensors at window end.
    Used as 'raw model score' without calibration.
    """
    last = values[:, -5:]
    z = (last - last.mean(axis=1, keepdims=True)) / (last.std(axis=1, keepdims=True) + 1e-6)
    return float(np.abs(z).mean())


def weak_decide(score: float, threshold: float = 0.5) -> WeakDecision:
    """Fixed threshold + immediate auto-trip on alarm (no dwell, no hysteresis)."""
    if score >= threshold:
        return WeakDecision(action="alarm", score=score, auto_tripped=True)
    if score >= threshold * 0.6:
        return WeakDecision(action="advisory", score=score, auto_tripped=False)
    return WeakDecision(action="nominal", score=score, auto_tripped=False)


def run_weak_pipeline(
    df: pd.DataFrame,
    threshold: float = 0.5,
    sensors: list[str] | None = None,
) -> tuple[list[WeakDecision], WindowBatch]:
    """Full weak path: forward-fill -> windows (no mask used) -> fixed threshold -> auto-trip."""
    filled = forward_fill_only(df, sensors)
    batch = make_windows(filled, sensors=sensors)
    decisions = [weak_decide(weak_score_from_window(w), threshold) for w in batch.values]
    return decisions, batch


def summarize_weak(decisions: list[WeakDecision]) -> dict[str, float]:
    actions = [d.action for d in decisions]
    return {
        "n_alarm": actions.count("alarm"),
        "n_advisory": actions.count("advisory"),
        "n_nominal": actions.count("nominal"),
        "n_auto_trip": sum(d.auto_tripped for d in decisions),
        "alarm_rate": actions.count("alarm") / max(len(actions), 1),
    }
