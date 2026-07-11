"""HAI raw feature reader.

The reader loads raw feature rows only. Labels are read by the final evaluator,
not by the protocol builder.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import RawDataset

HAI_NON_FEATURE_COLUMNS = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}


def read_hai_raw(data_dir: str | Path, max_train_rows: int | None = None) -> RawDataset:
    data_dir = Path(data_dir)
    train_paths = sorted(glob.glob(str(data_dir / "train*.csv.gz")))
    test_paths = sorted(glob.glob(str(data_dir / "test*.csv.gz")))
    if not train_paths:
        raise FileNotFoundError(f"no train*.csv.gz found in {data_dir}")
    if not test_paths:
        raise FileNotFoundError(f"no test*.csv.gz found in {data_dir}")

    train_df = pd.concat(
        (pd.read_csv(p, compression="gzip") for p in train_paths),
        ignore_index=True,
    )
    test_dfs = [pd.read_csv(p, compression="gzip") for p in test_paths]
    test_df = pd.concat(test_dfs, ignore_index=True)
    normal_df = train_df.reset_index(drop=True)
    if max_train_rows is not None and len(normal_df) > max_train_rows:
        normal_df = normal_df.iloc[:max_train_rows].reset_index(drop=True)

    feature_names = tuple(c for c in normal_df.columns if c not in HAI_NON_FEATURE_COLUMNS)
    normal_values = normal_df.loc[:, feature_names].fillna(0).astype(np.float32).to_numpy()
    test_values = test_df.loc[:, feature_names].fillna(0).astype(np.float32).to_numpy()
    test_value_parts = tuple(
        part.loc[:, feature_names].fillna(0).astype(np.float32).to_numpy()
        for part in test_dfs
    )
    normal_ts = (
        normal_df["time"].to_numpy()
        if "time" in normal_df.columns
        else np.arange(len(normal_df), dtype=np.int64)
    )
    test_ts = (
        test_df["time"].to_numpy()
        if "time" in test_df.columns
        else np.arange(len(test_df), dtype=np.int64)
    )
    test_ts_parts = tuple(
        part["time"].to_numpy()
        if "time" in part.columns
        else np.arange(len(part), dtype=np.int64)
        for part in test_dfs
    )
    return RawDataset(
        normal_values=normal_values,
        normal_timestamps=normal_ts,
        test_values=test_values,
        test_timestamps=test_ts,
        feature_names=feature_names,
        test_value_parts=test_value_parts,
        test_timestamp_parts=test_ts_parts,
    )
