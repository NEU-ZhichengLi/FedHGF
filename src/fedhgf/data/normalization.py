"""Train-only normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    std: np.ndarray
    fit_scope: str = "train"

    @classmethod
    def fit(cls, train_x: np.ndarray) -> "TrainOnlyStandardizer":
        mean = train_x.mean(axis=(0, 1), keepdims=True)
        std = train_x.std(axis=(0, 1), keepdims=True) + 1e-8
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

