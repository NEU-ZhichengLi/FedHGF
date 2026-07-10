"""
data_utils — Fair Benchmark 数据工具函数
Shared normalization and validation utilities for the FedHGF data loaders.
"""
from __future__ import annotations

import numpy as np


def normalize_client_data(client: dict) -> dict:
    """Per-node per-channel normalization using training split only."""
    X_tr = client["X_train"]
    mu   = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8

    out = dict(client)
    out["X_train"] = ((client["X_train"] - mu) / std).astype(np.float32)
    out["X_cal"]   = ((client["X_cal"]   - mu) / std).astype(np.float32)
    out["X_test"]  = ((client["X_test"]  - mu) / std).astype(np.float32)
    out["norm_mu"]  = mu.astype(np.float32)
    out["norm_std"] = std.astype(np.float32)
    return out


def validate_federation_clients(clients: list, n_anchor: int) -> None:
    if not clients:
        raise ValueError("clients must not be empty")

    required = {
        "client_name", "feature_names", "anchor_names", "aux_names",
        "X_train", "y_train", "X_cal", "y_cal", "X_test", "y_test",
        "n_k", "cal_anomaly_ratio", "total_test_anom", "is_sparse_anom",
        "norm_mu", "norm_std",
    }

    for idx, client in enumerate(clients):
        missing = required - set(client.keys())
        if missing:
            raise ValueError(f"client[{idx}] missing keys: {sorted(missing)}")

        for name in ("X_train", "X_cal", "X_test"):
            arr = client[name]
            if not isinstance(arr, np.ndarray) or arr.ndim != 4:
                raise ValueError(f"client[{idx}] {name} must be a 4D ndarray")
            if arr.dtype != np.float32:
                raise ValueError(f"client[{idx}] {name} must be float32")

        for name in ("y_train", "y_cal", "y_test"):
            arr = client[name]
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError(f"client[{idx}] {name} must be a 1D ndarray")
            if arr.dtype != np.int64:
                raise ValueError(f"client[{idx}] {name} must be int64")

        x_tr = client["X_train"]
        if x_tr.shape[2] != client["n_k"]:
            raise ValueError(f"client[{idx}] n_k does not match feature dimension")

        if len(client["feature_names"]) != client["n_k"]:
            raise ValueError(f"client[{idx}] feature_names length does not match n_k")

        if len(client["anchor_names"]) + len(client["aux_names"]) != client["n_k"]:
            raise ValueError(f"client[{idx}] anchor/aux split does not match n_k")

        if len(client["anchor_names"]) != n_anchor:
            raise ValueError(
                f"client[{idx}] anchor count ({len(client['anchor_names'])}) "
                f"!= n_anchor ({n_anchor})")

        y_test = client["y_test"]
        if int(client["total_test_anom"]) != int(y_test.sum()):
            raise ValueError(f"client[{idx}] total_test_anom != y_test.sum()")
