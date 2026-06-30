"""Calibration metrics and temperature scaling."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average |acc - conf| across bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, mean confidence, mean accuracy per bin."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, confs, accs = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        confs.append(probs[mask].mean())
        accs.append(labels[mask].mean())
    return np.array(centers), np.array(confs), np.array(accs)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax with temperature T."""
    scaled = logits / max(temperature, 1e-6)
    exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    init_t: float = 1.0,
) -> float:
    """Find the temperature T that minimizes NLL on a calibration set.

    Temperature scaling is a post-hoc calibration method: it divides the
    logits by a single scalar T before the softmax. T > 1 softens the
    distribution (reducing overconfidence), while T < 1 sharpens it. Because
    a monotonic rescaling of the logits does not change the argmax, the
    model's accuracy is unaffected -- only the confidence is recalibrated.

    The optimal T is the one that minimizes the negative log-likelihood (NLL)
    of the true labels under the temperature-scaled probabilities, which we
    solve for with a bounded scalar optimization.

    Args:
        logits: Raw model logits, either shape (n, 2) for two-class output or
            shape (n,) for a single hazard score. A 1-D array is expanded to
            two classes as [-logit, logit].
        labels: Integer class labels of shape (n,).
        init_t: Initial guess for the temperature passed to the optimizer.

    Returns:
        The fitted temperature as a float, constrained to [0.05, 10.0].
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        # Guard against passing probabilities instead of logits: probabilities
        # all live in [0, 1], whereas genuine logits should span that range.
        if np.all((logits >= 0.0) & (logits <= 1.0)):
            raise ValueError(
                "fit_temperature expects raw logits (n, 2) or hazard logits (n,), "
                "not probabilities. Use predict_logits(), not predict_proba()."
            )
        logits = np.stack([-logits, logits], axis=-1)

    def nll(t_arr: np.ndarray) -> float:
        # Clamp T away from zero to keep apply_temperature numerically stable.
        t = max(float(t_arr[0]), 1e-3)
        probs = apply_temperature(logits, t)
        # Pick out the probability assigned to each sample's true label.
        p = probs[np.arange(len(labels)), labels.astype(int)]
        return float(-np.log(np.clip(p, 1e-8, 1.0)).mean())

    result = minimize(nll, x0=[init_t], bounds=[(0.05, 10.0)])
    return float(result.x[0])


def calibrate_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated P(hazard=1) from binary logits (n, 2) or scores (n,)."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        # single hazard logit -> two-class
        logits = np.stack([-logits, logits], axis=-1)
    probs = apply_temperature(logits, temperature)
    return probs[:, 1]
