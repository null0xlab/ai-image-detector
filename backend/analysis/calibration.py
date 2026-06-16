"""Probability calibration for raw model scores (Platt scaling on 0-100 scale)."""

from __future__ import annotations

import math


def platt_calibrate(raw_score: float) -> float:
    """
    Map raw 0-100 forensic score to calibrated 0-100 (monotonic: higher raw -> higher out).
    Uses logistic on normalized input.
    """
    x = max(0.0, min(100.0, float(raw_score))) / 100.0
    # logit increases with x; center ~0.45 raw -> ~50% out
    logit = 5.2 * x - 2.1
    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(prob * 100.0, 2)


def calibrate_ensemble_scores(scores: dict[str, float]) -> dict[str, float]:
    return {name: platt_calibrate(score) for name, score in scores.items()}


def synthetic_and_authentic_likelihood(synthetic_score: float) -> dict[str, float]:
    """
    Expose clear semantics for UI:
    - synthetic_likelihood: 0–100, higher = more likely AI/synthetic
    - authentic_likelihood: 0–100, complement (not probabilistically calibrated)
    """
    syn = round(min(max(float(synthetic_score), 0.0), 100.0), 1)
    auth = round(100.0 - syn, 1)
    return {"synthetic_likelihood": syn, "authentic_likelihood": auth}
