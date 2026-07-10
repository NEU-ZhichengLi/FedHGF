"""
HAI 21.03 联邦数据加载器
========================
数据集：Hardware-In-the-Loop Augmented ICS Security Dataset (HAI 21.03)
来源：ETRI, https://github.com/icsdataset/hai

数据结构
--------
- 79 个传感器特征，按 4 个物理子进程自然划分：
    P1: 38 feats  (锅炉/热交换器)  ← 独立客户端
    P2: 22 feats  (汽轮机)         ← 独立客户端
    P3:  7 feats  (水箱/给水)      ← 升为全局共享锚点（参照 WADI zone 设计）
    P4: 12 feats  (蒸汽轮机)       ← 独立客户端

锚点设计（anchor_process="P3"，参照 WADI）
-----------------------------------------
P3（水箱）是整个热工系统的水源，其水位/流量对 P1/P2/P4 均有直接影响，
物理上等价于 WADI 中跨 zone 的全局管网传感器。

  Client P1: [P3 anchor(7)] + [P1 feats(38)] = 45 feats (7 anchor + 38 aux)
  Client P2: [P3 anchor(7)] + [P2 feats(22)] = 29 feats (7 anchor + 22 aux)
  Client P4: [P3 anchor(7)] + [P4 feats(12)] = 19 feats (7 anchor + 12 aux)
  n_anchor  = 7,  het_ratio = (45-19)/45 = 57.8%
  （介于 WADI Moderate 48.8% 与 Severe 71.8% 之间）

- 训练文件: train1/2/3.csv.gz（过滤 attack==0 行作为正常训练数据）
- 测试文件: test1/2/3/4/5.csv.gz（含 per-process 攻击标签）
- 攻击标签: attack（全局）/ attack_P1 / attack_P2 / attack_P3
  注意: P4 无独立攻击标签，使用全局 attack 列

接口
----
load_hai_federation(data_dir, ...) → (clients, n_anchor, anchor_cols)
  clients    : List[dict]，P1/P2/P4 各一个 client（P3 作锚点不单独成 client）
  n_anchor   : 7（P3 特征数，跨 client 共享锚点，激活 FedGAD 图通信）
  anchor_cols: P3 特征列名

用法示例
--------
  clients, n_anchor, _ = load_hai_federation("Data/HAI 21.03")
  # clients[0] → P1 (7 anchor + 38 aux = 45 feats)
  # clients[1] → P2 (7 anchor + 22 aux = 29 feats)
  # clients[2] → P4 (7 anchor + 12 aux = 19 feats)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_utils import normalize_client_data, validate_federation_clients

                                                                          

PROCESS_PREFIXES = ["P1", "P2", "P3", "P4"]
ANCHOR_PROCESS   = "P3"                                       
CLIENT_PROCESSES = ["P1", "P2", "P4"]                   

                                    
PROCESS_ATTACK_COL = {
    "P1": "attack_P1",
    "P2": "attack_P2",
    "P3": "attack_P3",
    "P4": "attack",                            
}

META_COLS = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}


                                                                          

def _windowize(
    values: np.ndarray,
    labels: np.ndarray,
    window_len: int,
    stride: int,
    label_mode: str = "any",
) -> Tuple[np.ndarray, np.ndarray]:
    """滑窗切片。values: [N, n_k, 1] → X: [N_win, T, n_k, 1]，y: [N_win]"""
    X, y = [], []
    n = len(values)
    for start in range(0, n - window_len + 1, stride):
        end = start + window_len
        X.append(values[start:end])
        chunk = labels[start:end]
        if label_mode == "last":
            y.append(int(chunk[-1] > 0))
        elif label_mode == "center":
            y.append(int(chunk[window_len // 2] > 0))
        elif label_mode == "majority":
            y.append(int(chunk.sum() > window_len // 2))
        else:         
            y.append(int(chunk.max() > 0))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _load_csvgz_files(paths: List[str]) -> pd.DataFrame:
    """加载并合并多个 .csv.gz 文件。"""
    dfs = []
    for p in sorted(paths):
        df = pd.read_csv(p, compression="gzip")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _get_process_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """获取指定进程前缀的所有特征列（排除 meta 列）。"""
    return [c for c in df.columns if c.startswith(prefix + "_") and c not in META_COLS]


                                                                           

def load_hai_federation(
    data_dir: str,
    processes: Optional[List[str]] = None,
    anchor_process: str = ANCHOR_PROCESS,
    window_len: int = 16,
    stride: int = 4,
    cal_anom_ratio: float = 0.05,
    cal_normal_frac: float = 0.15,
    test_anom_ratio: float = 0.15,
    max_train_rows: Optional[int] = 200_000,
    seed: int = 42,
    min_cal_anom: int = 20,
    label_mode: str = "any",
) -> Tuple[List[dict], int, List[str]]:
    """
    加载 HAI 21.03 数据集，按物理子进程构建异构联邦 clients。
    P3（水箱）默认作为全局共享锚点，不单独成 client。

    Parameters
    ----------
    data_dir       : HAI 21.03 数据目录（含 train*.csv.gz, test*.csv.gz）
    processes      : 独立客户端进程列表，默认 ["P1","P2","P4"]
    anchor_process : 用作全局共享锚点的进程，默认 "P3"
    window_len     : 滑窗长度（时间步数）
    stride         : 滑窗步长
    cal_anom_ratio : 攻击窗口中用于校准集的比例
    cal_normal_frac: 正常训练数据中留作校准的比例
    test_anom_ratio: 目标测试集异常比例（用于重采样平衡）
    max_train_rows : 最大训练行数（None=全量）
    seed           : 随机种子
    min_cal_anom   : 校准集最少异常窗口数

    Returns
    -------
    clients       : List[dict]，P1/P2/P4 各一个 client
    n_anchor      : len(p3_cols)（P3 特征数，共享锚点）
    anchor_cols   : p3_cols（P3 特征列名列表）
    """
    if processes is None:
        processes = list(CLIENT_PROCESSES)                           
                        
    client_processes = [p for p in processes if p != anchor_process]

    data_dir = Path(data_dir)
    rng = np.random.RandomState(seed)

    print(f"  [HAI] 加载训练文件...")
    train_paths = sorted(glob.glob(str(data_dir / "train*.csv.gz")))
    if not train_paths:
        raise FileNotFoundError(f"未找到 train*.csv.gz in {data_dir}")

    train_df = _load_csvgz_files(train_paths)
                           
    train_df = train_df[train_df["attack"] == 0].reset_index(drop=True)
    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.iloc[:max_train_rows].reset_index(drop=True)

    print(f"  [HAI] 训练行数（正常）: {len(train_df)}")

    print(f"  [HAI] 加载测试文件...")
    test_paths = sorted(glob.glob(str(data_dir / "test*.csv.gz")))
    if not test_paths:
        raise FileNotFoundError(f"未找到 test*.csv.gz in {data_dir}")

    test_df = _load_csvgz_files(test_paths)
    print(f"  [HAI] 测试行数: {len(test_df)}, "
          f"全局攻击帧: {int(test_df['attack'].sum())} "
          f"({test_df['attack'].mean():.2%})")

                                                                  
    p3_cols = _get_process_cols(train_df, anchor_process)
    if not p3_cols:
        raise ValueError(f"锚点进程 '{anchor_process}' 在训练数据中无特征列")
    n_anchor = len(p3_cols)
    print(f"  [HAI] 锚点进程={anchor_process}, n_anchor={n_anchor}: {p3_cols}")

    clients = []

    for proc in client_processes:
        if proc not in PROCESS_PREFIXES:
            raise ValueError(f"未知进程: {proc}，可选: {PROCESS_PREFIXES}")

        own_cols = _get_process_cols(train_df, proc)
        if not own_cols:
            raise ValueError(f"进程 {proc} 在训练数据中无特征列")
        feat_cols = p3_cols + own_cols                      

        attack_col = PROCESS_ATTACK_COL.get(proc, "attack")
        if attack_col not in test_df.columns:
            print(f"  [HAI/{proc}] ⚠ 攻击标签列 '{attack_col}' 不存在，退用全局 'attack'")
            attack_col = "attack"

        n_k = len(feat_cols)

                                                                
        X_tr_raw = train_df[feat_cols].fillna(0).astype(np.float32).to_numpy()
        y_tr_raw = np.zeros(len(X_tr_raw), dtype=np.int64)
        X_tr_raw = X_tr_raw[:, :, np.newaxis]               

        split = int(round(len(X_tr_raw) * (1.0 - cal_normal_frac)))
        split = min(max(split, window_len), len(X_tr_raw) - window_len)
        X_train, _ = _windowize(
            X_tr_raw[:split], np.zeros(split, dtype=np.int64),
            window_len, stride, label_mode)
        X_cal, y_cal = _windowize(
            X_tr_raw[split:], np.zeros(len(X_tr_raw) - split, dtype=np.int64),
            window_len, stride, label_mode)

        X_te_raw = test_df[feat_cols].fillna(0).astype(np.float32).to_numpy()
        y_te_raw = (test_df[attack_col].fillna(0).astype(int) > 0).astype(np.int64).to_numpy()
        X_te_raw = X_te_raw[:, :, np.newaxis]
        X_test, y_test = _windowize(X_te_raw, y_te_raw, window_len, stride, label_mode)

        client = {
            "client_name":       proc,
            "feature_names":     feat_cols,
            "anchor_names":      list(p3_cols),
            "aux_names":         own_cols,
            "X_train":           X_train,
            "y_train":           np.zeros(len(X_train), dtype=np.int64),
            "X_cal":             X_cal,
            "y_cal":             y_cal,
            "X_test":            X_test,
            "y_test":            y_test,
            "n_k":               n_k,
            "cal_anomaly_ratio": float(y_cal.mean()) if len(y_cal) > 0 else 0.0,
            "total_test_anom":   int(y_test.sum()),
            "is_sparse_anom":    int(y_test.sum()) < 20,
            "split_protocol":    "paper_chronological_label_free",
        }
        clients.append(normalize_client_data(client))

        print(
            f"  [HAI/{proc}] n_k={n_k}, train={len(X_train)}, "
            f"cal={len(X_cal)}(label-free), "
            f"test={len(X_test)}({int(y_test.sum())} anom, natural)"
        )
        continue

        X_all_win, _ = _windowize(X_tr_raw, y_tr_raw, window_len, stride, label_mode)
        n_all = len(X_all_win)
        n_train = max(256, int(n_all * (1.0 - cal_normal_frac)))
        n_train = min(n_train, n_all - 64) if n_all > 512 else max(1, int(n_all * 0.8))
        X_train = X_all_win[:n_train]
        X_cal_norm = X_all_win[n_train:].copy()

                                                                 
        X_te_raw = test_df[feat_cols].fillna(0).astype(np.float32).to_numpy()
        y_te_raw = (test_df[attack_col].fillna(0).astype(int) > 0).astype(np.int64).to_numpy()
        X_te_raw = X_te_raw[:, :, np.newaxis]

        X_attack_win, y_attack_win = _windowize(X_te_raw, y_te_raw, window_len, stride, label_mode)

        anom_mask = y_attack_win == 1
        X_anom = X_attack_win[anom_mask]
        X_norm_pool = X_attack_win[~anom_mask].copy()

        if len(X_anom) < min_cal_anom + 16:
            print(f"  [HAI/{proc}] ⚠ 攻击窗口不足 ({len(X_anom)})，跳过该进程")
            continue

                                                  
                                                 
        rng.shuffle(X_anom)
        n_cal_anom_target = max(
            min_cal_anom,
            int(round(len(X_cal_norm) * cal_anom_ratio / max(1e-8, 1.0 - cal_anom_ratio)))
        )
        n_cal_anom = min(n_cal_anom_target, len(X_anom) - max(8, len(X_anom) // 4))
        n_cal_anom = max(min_cal_anom, n_cal_anom)
        X_cal_anom = X_anom[:n_cal_anom]
        X_test_anom = X_anom[n_cal_anom:]

                                
        if len(X_norm_pool) >= 32:
            X_test_norm = X_norm_pool
        else:
            half = len(X_cal_norm) // 2
            X_test_norm = X_cal_norm[half:].copy()
            X_cal_norm = X_cal_norm[:half].copy()

                       
        n_anom_t = len(X_test_anom)
        n_norm_t = len(X_test_norm)
        if n_anom_t > 0 and n_norm_t > 0 and 0 < test_anom_ratio < 1:
            actual = n_anom_t / (n_anom_t + n_norm_t)
            if actual > test_anom_ratio + 0.05:
                n_keep = max(16, int(n_norm_t * test_anom_ratio / max(1e-8, 1 - test_anom_ratio)))
                X_test_anom = X_test_anom[:min(n_keep, n_anom_t)]
            elif actual < test_anom_ratio - 0.05:
                n_keep = max(16, int(n_anom_t * (1 - test_anom_ratio) / max(1e-8, test_anom_ratio)))
                rng.shuffle(X_test_norm)
                X_test_norm = X_test_norm[:min(n_keep, n_norm_t)]

               
        X_cal = np.concatenate([X_cal_norm, X_cal_anom], axis=0)
        y_cal = np.concatenate([
            np.zeros(len(X_cal_norm), dtype=np.int64),
            np.ones(len(X_cal_anom), dtype=np.int64),
        ])
        perm = rng.permutation(len(X_cal))
        X_cal, y_cal = X_cal[perm], y_cal[perm]

        X_test = np.concatenate([X_test_norm, X_test_anom], axis=0)
        y_test = np.concatenate([
            np.zeros(len(X_test_norm), dtype=np.int64),
            np.ones(len(X_test_anom), dtype=np.int64),
        ])
        perm = rng.permutation(len(X_test))
        X_test, y_test = X_test[perm], y_test[perm]

        client = {
            "client_name":       proc,
            "feature_names":     feat_cols,
            "anchor_names":      list(p3_cols),
            "aux_names":         own_cols,
            "X_train":           X_train,
            "y_train":           np.zeros(len(X_train), dtype=np.int64),
            "X_cal":             X_cal,
            "y_cal":             y_cal,
            "X_test":            X_test,
            "y_test":            y_test,
            "n_k":               n_k,
            "cal_anomaly_ratio": float(y_cal.mean()) if len(y_cal) > 0 else 0.0,
            "total_test_anom":   int(y_test.sum()),
            "is_sparse_anom":    int(y_test.sum()) < 20,
        }
        clients.append(normalize_client_data(client))

        print(
            f"  [HAI/{proc}] n_k={n_k}, train={len(X_train)}, "
            f"cal={len(X_cal)}({int(y_cal.sum())} anom, ratio={float(y_cal.mean()):.3f}), "
            f"test={len(X_test)}({int(y_test.sum())} anom, ratio={float(y_test.mean()):.3f})"
        )

    if not clients:
        raise ValueError("未能构建任何有效 client，请检查数据和参数")

    validate_federation_clients(clients, n_anchor)
    nk_list = [c["n_k"] for c in clients]
    het = (max(nk_list) - min(nk_list)) / max(nk_list)
    print(f"  [HAI] 共 {len(clients)} clients: {nk_list}, n_anchor={n_anchor}, het_ratio={het:.1%}")
    return clients, n_anchor, list(p3_cols)
