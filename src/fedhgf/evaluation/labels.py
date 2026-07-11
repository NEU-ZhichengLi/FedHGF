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


def load_test_labels(dataset: str, data_dir: str | Path) -> np.ndarray:
    data_dir = Path(data_dir)
    if dataset == "hai":
        return load_hai_test_labels(data_dir)
    if dataset == "wadi":
        test_df = pd.read_csv(data_dir / "WADI_attackdataLABLE.csv", header=1, low_memory=False)
        test_df.columns = [str(c).strip() for c in test_df.columns]
        label_cols = [
            c for c in test_df.columns
            if "attack" in c.lower() and ("label" in c.lower() or "lable" in c.lower())
        ]
        if not label_cols:
            raise ValueError("WADI test file does not contain an attack label column")
        raw = pd.to_numeric(test_df[label_cols[0]], errors="coerce").fillna(1)
        return (raw == -1).astype(np.int64).to_numpy()
    if dataset == "swat":
        test_df = pd.read_csv(data_dir / "merged.csv", usecols=["Normal/Attack"], low_memory=False)
        return (test_df["Normal/Attack"].astype(str).str.strip() == "Attack").astype(np.int64).to_numpy()
    if dataset == "batadal":
        test_df = pd.read_csv(data_dir / "BATADAL_dataset04.csv", skipinitialspace=True, usecols=lambda c: str(c).strip() == "ATT_FLAG")
        test_df.columns = [str(c).strip() for c in test_df.columns]
        raw = pd.to_numeric(test_df["ATT_FLAG"], errors="coerce").fillna(-999).astype(int)
        return (raw == 1).astype(np.int64).to_numpy()
    raise ValueError(f"unknown dataset: {dataset!r}")


def windowize_test_labels(
    dataset: str,
    data_dir: str | Path,
    window_length: int,
    stride: int,
    label_mode: str,
) -> np.ndarray:
    data_dir = Path(data_dir)
    if dataset == "hai":
        ys = []
        for path in sorted(glob.glob(str(data_dir / "test*.csv.gz"))):
            part = pd.read_csv(path, compression="gzip", usecols=["attack"])
            labels = (part["attack"].fillna(0).astype(int) > 0).astype(np.int64).to_numpy()
            ys.append(windowize_labels_for_evaluation(labels, window_length, stride, label_mode))
        return np.concatenate(ys) if ys else np.empty(0, dtype=np.int64)
    return windowize_labels_for_evaluation(
        load_test_labels(dataset, data_dir),
        window_length,
        stride,
        label_mode=label_mode,
    )


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
