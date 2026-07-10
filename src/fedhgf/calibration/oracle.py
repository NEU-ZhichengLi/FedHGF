"""Oracle calibration for diagnostics only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


class OracleF1Calibrator:
    def __init__(self) -> None:
        self.threshold_: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "OracleF1Calibrator":
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        best_f1 = -1.0
        best_tau = float(np.quantile(scores, 0.95))
        for q in np.linspace(0.01, 0.99, 99):
            tau = float(np.quantile(scores, q))
            pred = (scores > tau).astype(np.int64)
            f1 = float(f1_score(labels, pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        self.threshold_ = best_tau
        return self

