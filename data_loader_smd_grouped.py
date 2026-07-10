"""
SMD 联邦数据加载器（分组版 — 按服务器集群分 client）
=====================================================

设计逻辑
--------
  SMD 共 28 台服务器，分 3 个集群：machine-1 (8台), machine-2 (9台), machine-3 (11台)。
  同组机器共享相同的 38 维特征 schema (f0-f37)。

  分组方案：每个集群 = 1 个 federated client (K=3)
    - Client 数据 = 组内所有机器训练/测试数据纵向拼接
    - 所有 client 维度相同 (38维)
    - Anchor = 全局固定的 top-n 方差维度 (跨组平均)

  优点：
    1. K=3 与其他数据集量级一致
    2. 组内数据量大（~19-29万行），训练充分
    3. 组间异常模式天然不同（不同集群监控不同服务）
    4. 保持 anchor 全局一致语义

采样间隔：1 分钟
特征：f0-f37（CPU、内存、磁盘、网络等系统指标）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if str(_here.parent / "New") not in sys.path:
    sys.path.append(str(_here.parent / "New"))
from data_utils import normalize_client_data, validate_federation_clients


                        
_GROUP_NAMES = ["machine-1", "machine-2", "machine-3"]


def _load_machine(train_path: Path, test_path: Path, label_path: Path):
    X_tr = pd.read_csv(train_path, header=None).to_numpy(np.float32)
    X_te = pd.read_csv(test_path,  header=None).to_numpy(np.float32)
    y_te = pd.read_csv(label_path, header=None).to_numpy(np.int64).ravel()
    return X_tr, X_te, y_te


def _windowize(X: np.ndarray, y: np.ndarray,
               wlen: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    wins, labs = [], []
    for s in range(0, len(X) - wlen + 1, stride):
        wins.append(X[s: s + wlen])
        labs.append(int(y[s: s + wlen].max() > 0))
    if not wins:
        return np.empty((0, wlen) + X.shape[1:], np.float32), np.empty(0, np.int64)
    return np.asarray(wins, np.float32), np.asarray(labs, np.int64)


def compute_global_anchor_indices(
    data_dir: str,
    n_anchor: int = 8,
    exclude_machines: Optional[Iterable[str]] = None,
) -> List[int]:
    """跨所有机器计算平均训练集方差，返回 top-n_anchor 全局 feature 索引。"""
    base = Path(data_dir)
    train_dir = base / "train"
    excl = set(exclude_machines) if exclude_machines else set()

    all_vars: List[np.ndarray] = []
    for f in sorted(os.listdir(train_dir)):
        mname = Path(f).stem
        if mname in excl:
            continue
        X_tr = pd.read_csv(train_dir / f, header=None).to_numpy(np.float32)
        all_vars.append(X_tr.var(axis=0))

    if not all_vars:
        raise RuntimeError("No valid SMD machines found for anchor computation.")

    mean_var = np.mean(np.stack(all_vars, axis=0), axis=0)
    anchor_idx = list(np.argsort(-mean_var)[:n_anchor])
    return anchor_idx


def load_smd_grouped(
    data_dir: str,
    n_anchor: int = 8,
    window_len: int = 20,
    stride: int = 5,
    cal_normal_frac: float = 0.15,
    cal_anom_ratio: float = 0.10,
    test_anom_ratio: float = 0.10,
    min_cal_anom: int = 30,
    seed: int = 42,
    max_train_rows: int = 200000,
    exclude_machines: Optional[Iterable[str]] = None,
    anchor_indices: Optional[List[int]] = None,
    label_mode: str = None,                                 
) -> Tuple[List[dict], int, List[str]]:
    """
    SMD 分组加载器：每个服务器集群 = 1 个联邦 client。

    返回: (clients, n_anchor, anchor_feature_names)
    """
    rng = np.random.RandomState(seed)
    base = Path(data_dir)
    train_dir = base / "train"
    test_dir  = base / "test"
    label_dir = base / "test_label"

    excl_set = set(exclude_machines) if exclude_machines else set()

                     
    if anchor_indices is None:
        anchor_indices = compute_global_anchor_indices(
            data_dir, n_anchor=n_anchor, exclude_machines=excl_set)
    effective_n = len(anchor_indices)

    n_dim = 38             
    all_idx = list(range(n_dim))
    aux_idx = [i for i in all_idx if i not in set(anchor_indices)]
    feat_idx = anchor_indices + aux_idx                           
    anchor_names = [f"f{i}" for i in anchor_indices]
    aux_names = [f"f{i}" for i in aux_idx]
    n_k = len(feat_idx)

    print(f"  [SMD-grouped] n_anchor={effective_n}, anchor_idx={anchor_indices}")
    print(f"  [SMD-grouped] n_dim={n_dim}, aux={len(aux_idx)}")

                     
    all_files = sorted(os.listdir(train_dir))
    group_files: Dict[str, List[str]] = {g: [] for g in _GROUP_NAMES}
    for f in all_files:
        stem = Path(f).stem
        if stem in excl_set:
            continue
        for g in _GROUP_NAMES:
            if stem.startswith(g):
                group_files[g].append(stem)
                break

                      
    clients: List[dict] = []
    for group_name, machines in group_files.items():
        if not machines:
            continue

                    
        X_tr_list, X_te_list, y_te_list = [], [], []
        for m in machines:
            xtr, xte, yte = _load_machine(
                train_dir / f"{m}.txt", test_dir / f"{m}.txt", label_dir / f"{m}.txt")
            X_tr_list.append(xtr)
            X_te_list.append(xte)
            y_te_list.append(yte)

        X_tr_raw = np.concatenate(X_tr_list, axis=0)
        X_te_raw = np.concatenate(X_te_list, axis=0)
        y_te_raw = np.concatenate(y_te_list, axis=0)

                 
        if len(X_tr_raw) > max_train_rows:
            idx = rng.choice(len(X_tr_raw), max_train_rows, replace=False)
            idx.sort()
            X_tr_raw = X_tr_raw[idx]

        print(f"  [{group_name}] machines={len(machines)}, "
              f"train={X_tr_raw.shape[0]}, test={X_te_raw.shape[0]}, "
              f"anom={int(y_te_raw.sum())} ({y_te_raw.mean()*100:.1f}%)")

                           
        X_tr_raw = X_tr_raw[:, feat_idx]
        X_te_raw = X_te_raw[:, feat_idx]

                   
        y_tr_raw = np.zeros(len(X_tr_raw), np.int64)

             
        X_tr_w, _ = _windowize(X_tr_raw, y_tr_raw, window_len, stride)
        X_te_w, y_te_w = _windowize(X_te_raw, y_te_raw, window_len, stride)
        X_tr_w = X_tr_w[..., None]
        X_te_w = X_te_w[..., None]

                   
        n_tr = len(X_tr_w)
        n_train = max(64, int(n_tr * (1.0 - cal_normal_frac)))
        n_train = min(n_train, n_tr - 64)
        X_train = X_tr_w[:n_train]
        X_hold  = X_tr_w[n_train:]

        n_cal_nrm = max(16, min(int(len(X_hold) * 0.5), len(X_hold) - 8))
        X_cal_nrm  = X_hold[:n_cal_nrm]
        X_test_nrm = X_hold[n_cal_nrm:]

                                  
        anom_m = y_te_w == 1
        X_anom = X_te_w[anom_m].copy()
        X_nrm_from_test = X_te_w[~anom_m].copy()
        total_anom = len(X_anom)

        if total_anom < min_cal_anom:
            X_cal_anom  = np.empty((0,) + X_train.shape[1:], np.float32)
            X_test_anom = X_anom
        else:
            rng.shuffle(X_anom)
            n_by_r = max(0, int(round(
                len(X_cal_nrm) * cal_anom_ratio / max(1e-8, 1 - cal_anom_ratio))))
            n_ca = max(min_cal_anom, min(n_by_r, total_anom - max(8, total_anom // 4)))
            X_cal_anom  = X_anom[:n_ca]
            X_test_anom = X_anom[n_ca:]

                
        X_cal = np.concatenate([X_cal_nrm, X_cal_anom])
        y_cal = np.concatenate([np.zeros(len(X_cal_nrm), np.int64),
                                np.ones(len(X_cal_anom), np.int64)])
        p = rng.permutation(len(X_cal)); X_cal, y_cal = X_cal[p], y_cal[p]

                                    
        X_test_nrm_pool = np.concatenate([X_nrm_from_test, X_test_nrm])
        rng.shuffle(X_test_nrm_pool)
        if test_anom_ratio and len(X_test_anom) > 0 and len(X_test_nrm_pool) > 0:
            n_test_norm_target = int(round(
                len(X_test_anom) * (1.0 - test_anom_ratio) / max(1e-8, test_anom_ratio)))
            n_test_norm = min(len(X_test_nrm_pool), max(32, n_test_norm_target))
            X_test_nrm_final = X_test_nrm_pool[:n_test_norm]
        else:
            X_test_nrm_final = X_test_nrm_pool

        X_test = np.concatenate([X_test_nrm_final, X_test_anom])
        y_test = np.concatenate([np.zeros(len(X_test_nrm_final), np.int64),
                                 np.ones(len(X_test_anom), np.int64)])
        p = rng.permutation(len(X_test)); X_test, y_test = X_test[p], y_test[p]

        feat_names = anchor_names + aux_names
        client = {
            "client_name":   group_name,
            "feature_names": feat_names,
            "anchor_names":  anchor_names,
            "aux_names":     aux_names,
            "anchor_indices_global": anchor_indices,
            "X_train": X_train,
            "y_train": np.zeros(len(X_train), np.int64),
            "X_cal":   X_cal,
            "y_cal":   y_cal,
            "X_test":  X_test,
            "y_test":  y_test,
            "n_k":     n_k,
            "cal_anomaly_ratio": float(y_cal.mean()) if len(y_cal) else 0.0,
            "total_test_anom":   int(y_test.sum()),
            "is_sparse_anom":    total_anom < min_cal_anom,
            "machines_in_group": machines,
        }
        clients.append(normalize_client_data(client))

    print(f"  [SMD-grouped] {len(clients)} clients (K={len(clients)}), "
          f"n_anchor={effective_n}, n_k={n_k}")
    validate_federation_clients(clients, effective_n)
    return clients, effective_n, anchor_names
