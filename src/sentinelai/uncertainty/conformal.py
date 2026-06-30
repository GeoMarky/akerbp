"""Split and Mondrian conformal prediction for binary hazard detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalPredictor:
    """
    Split conformal classifier producing prediction sets over {nominal, hazard}.

    Uses nonconformity score 1 - P(true class) on a calibration set.
    """

    alpha: float = 0.10
    q_hat: float = 0.0
    q_hat_nominal: float = 0.0
    q_hat_hazard: float = 0.0
    mondrian: bool = False

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        mondrian: bool | None = None,
    ) -> ConformalPredictor:
        """Fit threshold on calibration probabilities."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        self.mondrian = mondrian if mondrian is not None else self.mondrian

        scores = 1.0 - probs[np.arange(len(labels)), labels]
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)

        if self.mondrian:
            for cls, attr in [(0, "q_hat_nominal"), (1, "q_hat_hazard")]:
                mask = labels == cls
                if mask.sum() == 0:
                    setattr(self, attr, 1.0)
                    continue
                s = scores[mask]
                nc = len(s)
                ql = min(1.0, np.ceil((nc + 1) * (1 - self.alpha)) / nc)
                setattr(self, attr, float(np.quantile(s, ql, method="higher")))
        else:
            self.q_hat = float(np.quantile(scores, q_level, method="higher"))
        return self

    def predict_set(self, probs: np.ndarray) -> list[list[str]]:
        """
        Return prediction set per sample.
        probs: (n, 2) with [P(nominal), P(hazard)] or (n,) P(hazard).
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=-1)

        sets: list[list[str]] = []
        for p in probs:
            p_nom, p_haz = p[0], p[1]
            if self.mondrian:
                inc_nom = (1 - p_nom) <= self.q_hat_nominal
                inc_haz = (1 - p_haz) <= self.q_hat_hazard
            else:
                inc_nom = (1 - p_nom) <= self.q_hat
                inc_haz = (1 - p_haz) <= self.q_hat

            s: list[str] = []
            if inc_nom:
                s.append("nominal")
            if inc_haz:
                s.append("hazard")
            if not s:
                s = ["nominal", "hazard"]  # empty set -> abstain (ambiguous)
            sets.append(s)
        return sets

    def predict_set_labels(self, probs: np.ndarray) -> list[str]:
        """Collapse set to single label for metrics (both -> ambiguous)."""
        out = []
        for s in self.predict_set(probs):
            if s == ["nominal"]:
                out.append("nominal")
            elif s == ["hazard"]:
                out.append("hazard")
            else:
                out.append("ambiguous")
        return out
