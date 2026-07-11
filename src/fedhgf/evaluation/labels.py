"""Evaluation-only label loading and windowing."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_hai_test_labels(data_dir: str | Path) -> np.ndarray:
    data_dir = Path(data_dir)
    test_paths = sorted(glob.glob(str(data_dir / "test*.csv.gz")))
    if not test_paths:
        raise FileNotFoundError(f"no test*.csv.gz found in {data_dir}")
    test_df = pd.concat(
        (pd.read_csv(p, compression="gzip", usecols=["attack"]) for p in test_paths),
        ignore_index=True,
    )
    return (test_df["attack"].fillna(0).astype(int) > 0).astype(np.int64).to_numpy()


def windowize_labels_for_evaluation(
    labels: np.ndarray,
    window_length: int,
    stride: int,
    label_mode: str = "any",
) -> np.ndarray:
    y: list[int] = []
    labels = np.asarray(labels, dtype=np.int64)
    for start in range(0, len(labels) - window_length + 1, stride):
        end = start + window_length
        chunk = labels[start:end]
        if label_mode == "last":
            y.append(int(chunk[-1] > 0))
        elif label_mode == "center":
            y.append(int(chunk[window_length // 2] > 0))
        elif label_mode == "majority":
            y.append(int(chunk.sum() > window_length // 2))
        elif label_mode == "any":
            y.append(int(chunk.max() > 0))
        else:
            raise ValueError(f"unknown label_mode: {label_mode!r}")
    return np.asarray(y, dtype=np.int64)
