"""Label-free quantile calibration."""

from __future__ import annotations

import numpy as np


class QuantileCalibrator:
    def __init__(self, alert_budget: float = 0.05):
        if not 0.0 < alert_budget < 1.0:
            raise ValueError("alert_budget must be in (0, 1)")
        self.alert_budget = float(alert_budget)
        self.threshold_: float | None = None

    def fit(self, calibration_scores: np.ndarray) -> "QuantileCalibrator":
        scores = np.asarray(calibration_scores, dtype=np.float32)
        if scores.ndim != 1:
            raise ValueError("calibration_scores must be one-dimensional")
        if len(scores) == 0:
            raise ValueError("calibration_scores must not be empty")
        self.threshold_ = float(np.quantile(scores, 1.0 - self.alert_budget))
        return self

    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("fit must be called before predict")
        scores = np.asarray(test_scores, dtype=np.float32)
        return (scores > self.threshold_).astype(np.int64)

