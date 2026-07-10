"""
SMD 联邦数据加载器（Fair Benchmark 版 — 全局固定 anchor 设计）
=============================================================

问题：原设计每台机器独立选自己的 top-8 方差特征作 anchor。
      不同机器的 anchor-slot-i 可能对应完全不同的指标（CPU vs disk vs network）。
      FedHGF anchor graph 假设 slot-i 在所有客户端是可比的，该假设在原设计下不成立。

修复：全局固定 anchor index
-----------------------------------------------
  1. 对所有机器的训练集，计算每个维度的方差
  2. 取各机器方差的**算术平均值**
  3. 选取均值方差最大的 top-k 维度作为全局 anchor index
  4. 所有机器使用完全相同的 anchor 列索引

这样保证：anchor-slot-i 在所有客户端对应相同语义位置的系统指标。

数据集统计
----------
  来源：阿里云内部服务器指标（28台同类服务器，均为38维）
  采样间隔：1分钟
  特征：f0-f37（CPU使用率、内存、磁盘、网络IO等系统指标，schema一致）
  训练集：每台 ~28,479 行
  测试集：每台 ~28,479 行
  攻击比例：0.4% ~ 15.7%（均值 4.2%）

降级说明
--------
  SMD 被明确定位为 Tier 3 Functional-Alignment Benchmark。
  - Anchor 在不同机器间是"功能位置对齐"而非物理共享
  - SMD 主要用于验证 FedHGF 在同质化服务器场景的有效性
  - 不应声称 SMD anchor 具有与 WADI/HAI/BATADAL 相同的物理共享语义
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

_DEFAULT_EXCLUDE = ["machine-1-3", "machine-1-4"]


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
    """
    跨所有机器计算平均训练集方差，返回 top-n_anchor 的全局 feature 索引。
    这确保所有机器使用相同含义的 anchor 维度（功能对齐而非统计随机）。
    """
    base = Path(data_dir)
    train_dir = base / "train"
    excl = set(exclude_machines) if exclude_machines else set()

    all_vars: List[np.ndarray] = []
    for f in sorted(os.listdir(train_dir)):
        mname = Path(f).stem
        if any(mname == e or mname.startswith(e) for e in excl):
            continue
        X_tr = pd.read_csv(train_dir / f, header=None).to_numpy(np.float32)
        all_vars.append(X_tr.var(axis=0))                 

    if not all_vars:
        raise RuntimeError("No valid SMD machines found for anchor computation.")

    mean_var = np.mean(np.stack(all_vars, axis=0), axis=0)            
    anchor_idx = list(np.argsort(-mean_var)[:n_anchor])
    print(f"  [SMD fair] 全局 anchor 索引 (top-{n_anchor} 均值方差):")
    print(f"  {anchor_idx}  (对应特征名: {['f%d' % i for i in anchor_idx]})")
    return anchor_idx


def load_smd_fair(
    data_dir: str,
    n_anchor: int = 8,
    window_len: int = 20,
    stride: int = 5,
    cal_normal_frac: float = 0.15,
    cal_anom_ratio: float = 0.10,
    test_anom_ratio: float = 0.10,
    min_cal_anom: int = 20,
    seed: int = 42,
    max_clients: Optional[int] = None,
    exclude_machines: Optional[Iterable[str]] = None,
    anchor_indices: Optional[List[int]] = None,
) -> Tuple[List[dict], int, List[str]]:
    """
    构造 SMD 联邦基准（fair 版，全局固定 anchor index）。

    参数
    ----
    anchor_indices : 若 None，自动计算跨机器平均方差 top-n_anchor 索引。
                     可传入预计算结果以保证跨实验可重复性。

    返回
    ----
    (clients, n_anchor, anchor_feature_names)
    anchor_feature_names 形如 ['f6', 'f5', 'f10', ...] 在所有客户端含义一致。
    """
    rng = np.random.RandomState(seed)
    base = Path(data_dir)
    train_dir = base / "train"
    test_dir  = base / "test"
    label_dir = base / "test_label"

    excl_set = set(exclude_machines) if exclude_machines else set(_DEFAULT_EXCLUDE)

                                                                     
    if anchor_indices is None:
        anchor_indices = compute_global_anchor_indices(
            data_dir, n_anchor=n_anchor, exclude_machines=excl_set)
    effective_n = len(anchor_indices)

                                                                   
    all_files = sorted([
        f for f in os.listdir(train_dir)
        if Path(f).stem not in excl_set
        and not any(Path(f).stem.startswith(e) for e in excl_set)
    ])
    if max_clients:
        all_files = all_files[:max_clients]

    n_dim = pd.read_csv(train_dir / all_files[0], header=None).shape[1]
    all_idx  = list(range(n_dim))
    aux_idx  = [i for i in all_idx if i not in set(anchor_indices)]
    anchor_names = [f"f{i}" for i in anchor_indices]
    aux_names_template = [f"f{i}" for i in aux_idx]

    print(f"  [SMD fair] n_dim={n_dim}, anchor={anchor_indices}, aux={len(aux_idx)}")

    clients: List[dict] = []
    for fname in all_files:
        stem = Path(fname).stem
        X_tr, X_te, y_te = _load_machine(
            train_dir / fname, test_dir / fname, label_dir / fname)

                                                                
        feat_idx = anchor_indices + aux_idx
        X_tr = X_tr[:, feat_idx]
        X_te = X_te[:, feat_idx]
        n_k = len(feat_idx)

        y_tr = np.zeros(len(X_tr), np.int64)

        X_tr_w, _ = _windowize(X_tr[:, :, None], y_tr, window_len, stride)
        X_te_w, y_te_w = _windowize(X_te[:, :, None], y_te, window_len, stride)

        if len(X_tr_w) < 128 or len(X_te_w) < 32:
            print(f"  [SMD] {stem}: 窗口不足，跳过")
            continue

                                                                       
        n_tr = len(X_tr_w)
        n_train = max(64, int(n_tr * (1.0 - cal_normal_frac)))
        n_train = min(n_train, n_tr - 64)
        X_train = X_tr_w[:n_train]
        X_hold  = X_tr_w[n_train:]

        n_cal_nrm = max(16, min(int(len(X_hold) * 0.4), len(X_hold) - 8))
        X_cal_nrm  = X_hold[:n_cal_nrm]
        X_test_nrm = X_hold[n_cal_nrm:]

                                                                       
        anom_m = y_te_w == 1
        X_anom = X_te_w[anom_m].copy()
        X_test_a_full = X_te_w[~anom_m].copy()              
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

        X_cal  = np.concatenate([X_cal_nrm, X_cal_anom])
        y_cal  = np.concatenate([np.zeros(len(X_cal_nrm),  np.int64),
                                  np.ones(len(X_cal_anom),  np.int64)])
        p = rng.permutation(len(X_cal)); X_cal, y_cal = X_cal[p], y_cal[p]

                                                               
        X_test_nrm_pool = np.concatenate([X_test_a_full, X_test_nrm])
        rng.shuffle(X_test_nrm_pool)
        if test_anom_ratio and len(X_test_anom) > 0 and len(X_test_nrm_pool) > 0:
            n_test_norm_target = int(round(
                len(X_test_anom) * (1.0 - test_anom_ratio)
                / max(1e-8, test_anom_ratio)
            ))
            n_test_norm = min(len(X_test_nrm_pool), max(32, n_test_norm_target))
            X_test_nrm_final = X_test_nrm_pool[:n_test_norm]
        else:
            X_test_nrm_final = X_test_nrm_pool

        X_test = np.concatenate([X_test_nrm_final, X_test_anom])
        y_test = np.concatenate([np.zeros(len(X_test_nrm_final), np.int64),
                                  np.ones(len(X_test_anom), np.int64)])
        p = rng.permutation(len(X_test)); X_test, y_test = X_test[p], y_test[p]

        feat_names = anchor_names + aux_names_template
        client = {
            "client_name":   stem,
            "feature_names": feat_names,
            "anchor_names":  anchor_names,
            "aux_names":     aux_names_template,
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
        }
        clients.append(normalize_client_data(client))

    print(f"  [SMD fair] {len(clients)} clients, n_anchor={effective_n} "
          f"(global fixed index), each n_k={n_dim}")
    validate_federation_clients(clients, effective_n)
    return clients, effective_n, anchor_names
