"""
WADI 联邦数据加载器（Fair Benchmark 版 — zone3 物理锚点设计）
=============================================================

数据集统计
----------
  正常数据：WADI_14days_new.csv     约 1,209,601 行（14天，1秒采样）
  攻击数据：WADI_attackdataLABLE.csv  172,803 行，其中攻击 9,977 行（5.8%）
  总特征数：127 个传感器（去除元数据列后）

传感器区域划分
-------------
  zone1 (1_*) : 19 个传感器 — 进水及初级处理
  zone2 (2_*) : 82 个传感器 — 主配水网络（含 2A_*/2B_* 支路各4个）
  zone3 (3_*) : 15 个传感器 — 储水罐及二次配水
  global      :  3 个传感器 — LEAK_DIFF_PRESSURE, PLANT_START_STOP_LOG,
                               TOTAL_CONS_REQUIRED_FLOW

zone3 传感器（15个，均为全厂共享监控指标）：
  AIT类 (水质): 3_AIT_001_PV ~ 3_AIT_005_PV  (5个)
  流量计:        3_FIT_001_PV                  (1个)
  液位开关/传感器: 3_LS_001_AL, 3_LT_001_PV   (2个)
  阀门状态:      3_MV_001_STATUS ~ 3_MV_003_STATUS (3个)
  泵状态:        3_P_001_STATUS ~ 3_P_004_STATUS   (4个)

为何 zone3 适合作为全局 anchor
-------------------------------
WADI 系统物理拓扑：
  zone1（进水）→ zone2（配水）→ zone3（储水罐）→ 终端用户

zone3 在物理上是 zone1 和 zone2 的公共下游：
  - zone3 的储水罐 LT_001 反映整个系统的供水平衡
  - zone1 和 zone2 的控制器均通过 SCADA 读取 zone3 液位/流量进行前馈控制
  - zone3 的阀门/泵状态变化是全网异常的汇聚点

这与 HAI 数据集中以 P3（水箱）作为全局锚点进程完全对称：
  HAI: P1/P2 控制器观测 P3 水箱状态 → 相同设计原理
  WADI: zone1/zone2 控制器观测 zone3 储水状态

联邦划分结果
-----------
  anchor zone：zone3（15个传感器，选取 n_anchor 个作全局锚点）
  client zone1：zone1（19个）+ zone3 anchor（8个）= 27维
  client zone2：zone2（82个）+ zone3 anchor（8个）= 90维
  异构度：90 - 27 = 63，het_ratio = (90-27)/90 = 70%
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if str(_here.parent / "New") not in sys.path:
    sys.path.append(str(_here.parent / "New"))
from data_utils import normalize_client_data, validate_federation_clients

                                                                             
        
                                                                             
ZONE_PREFIXES: OrderedDict = OrderedDict([
    ("zone1",  "1_"),
    ("zone2",  "2_"),
    ("zone2a", "2A_"),                           
    ("zone2b", "2B_"),                           
    ("zone3",  "3_"),
])

GLOBAL_SENSOR_COLS = [
    "LEAK_DIFF_PRESSURE",
    "PLANT_START_STOP_LOG",
    "TOTAL_CONS_REQUIRED_FLOW",
]

_META_KEYWORDS = {"row", "date", "time", "attack", "lable", "label", "index"}

                                                                             
                                                     
       
                                         
                            
                          
                            
                          
                          
                              
                              
                               
                   
WADI_ZONE3_FIXED8 = [
    "3_LT_001_PV",
    "3_FIT_001_PV",
    "3_LS_001_AL",
    "3_MV_001_STATUS",
    "3_MV_002_STATUS",
    "3_MV_003_STATUS",
    "3_P_001_STATUS",
    "3_P_002_STATUS",
]


def _is_meta(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in _META_KEYWORDS)


def _classify(feat_cols: List[str]) -> Dict[str, List[str]]:
    """classify_features -> zone_map.
    zone2a / zone2b 列合并入 zone2（它们是 zone2 配水网络的支路）。
    """
    zone_map: Dict[str, List[str]] = {z: [] for z in ZONE_PREFIXES}
    zone_map["global"] = []
    for c in feat_cols:
        if c in GLOBAL_SENSOR_COLS:
            zone_map["global"].append(c)
            continue
        placed = False
        for zone_name, prefix in ZONE_PREFIXES.items():
            if c.startswith(prefix):
                zone_map[zone_name].append(c)
                placed = True
                break
        if not placed:
            zone_map["global"].append(c)
                                
    zone_map["zone2"] = (
        zone_map.pop("zone2", []) +
        zone_map.pop("zone2a", []) +
        zone_map.pop("zone2b", [])
    )
    return zone_map


def _windowize(X: np.ndarray, y: np.ndarray,
               window_len: int, stride: int,
               label_mode: str = "any") -> Tuple[np.ndarray, np.ndarray]:
    windows, labels = [], []
    n = len(X)
    for s in range(0, n - window_len + 1, stride):
        windows.append(X[s: s + window_len])
        chunk = y[s: s + window_len]
        if label_mode == "last":
            labels.append(int(chunk[-1] > 0))
        elif label_mode == "center":
            labels.append(int(chunk[window_len // 2] > 0))
        elif label_mode == "majority":
            labels.append(int(chunk.sum() > window_len // 2))
        else:         
            labels.append(int(chunk.max() > 0))
    if not windows:
        return np.empty((0, window_len) + X.shape[1:], dtype=np.float32), \
               np.empty(0, dtype=np.int64)
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_wadi_fair(
    data_dir: str,
    window_len: int = 16,
    stride: int = 4,
    cal_anom_ratio: float = 0.05,
    test_anom_ratio: Optional[float] = 0.15,
    cal_normal_frac: float = 0.15,
    seed: int = 42,
    n_anchor: int = 8,
    anchor_method: str = "variance",                          
    anchor_mode: str = "fixed8",                                         
    min_cal_anom: int = 20,
    max_train_rows: Optional[int] = 120_000,
    max_attack_rows: Optional[int] = None,
    max_train_windows: Optional[int] = None,
    mem_warn_mb: float = 500.0,
    include_global_in_anchor: bool = True,
    client_zones: Optional[List[str]] = None,                              
    label_mode: str = "any",
) -> Tuple[List[dict], int, List[str]]:
    """
    构造 WADI 联邦基准（fair 版，zone3 物理锚点设计）。

    参数
    ----
    include_global_in_anchor : True 时将 3 个全局传感器拼入每个 client
                               的 aux（不占 anchor 位置，额外信息）
    client_zones             : 参与联邦的客户端 zone 列表，默认 ["zone1", "zone2"]
                               zone3 始终作为 anchor 来源，不作为 client

    返回
    ----
    (clients, n_anchor, anchor_col_names)
    """
    rng = np.random.RandomState(seed)
    base = Path(data_dir)

                                                                       
    df_n = pd.read_csv(base / "WADI_14days_new.csv",
                       low_memory=False, nrows=max_train_rows)
    df_n.columns = [c.strip() for c in df_n.columns]
    df_n = df_n.ffill().bfill().fillna(0.0)

    df_a = pd.read_csv(base / "WADI_attackdataLABLE.csv",
                       header=1, low_memory=False, nrows=max_attack_rows)
    df_a.columns = [c.strip() for c in df_a.columns]
    df_a = df_a.ffill().bfill().fillna(0.0)

    feat_cols = [c for c in df_n.columns if not _is_meta(c)]

                                                                       
    zone_map = _classify(feat_cols)
    zone3_cols = zone_map.get("zone3", [])
    if len(zone3_cols) == 0:
        raise RuntimeError(
            "未找到 zone3 (3_*) 列，无法使用 zone3-anchor 设计。"
            f"当前数据集特征样例: {feat_cols[:5]}"
        )

                                                                         
    col2idx = {c: i for i, c in enumerate(feat_cols)}
    X_norm_full = df_n[feat_cols].to_numpy(dtype=np.float32)
    X_atk_full  = df_a[[c for c in feat_cols if c in df_a.columns]].to_numpy(
        dtype=np.float32)
                                        
    atk_col_map = {c: i for i, c in enumerate(df_a.columns)}
    X_atk_aligned = np.zeros((len(df_a), len(feat_cols)), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        if c in atk_col_map:
            X_atk_aligned[:, i] = df_a.iloc[:, atk_col_map[c]].to_numpy(
                dtype=np.float32)
    X_atk_full = X_atk_aligned

    lbl_col = [c for c in df_a.columns
               if "attack" in c.lower() and ("lable" in c.lower() or "label" in c.lower())]
    if not lbl_col:
        raise RuntimeError("未找到 WADI attack 标签列。")
    y_atk_raw  = pd.to_numeric(df_a[lbl_col[0]], errors="coerce").fillna(1)
    y_atk_full = (y_atk_raw == -1).to_numpy(dtype=np.int64)
    y_norm_full = np.zeros(len(X_norm_full), dtype=np.int64)

                                                                     
    if anchor_mode == "all15":
        anchor_names = [c for c in zone3_cols if c in col2idx]
        mode_desc = f"all-{len(anchor_names)} zone3 variables (no selection bias)"
    elif anchor_mode == "fixed8":
        anchor_names = [c for c in WADI_ZONE3_FIXED8 if c in col2idx]
        if len(anchor_names) < 4:
            print(f"  [WADI] fixed8 列名仅匹配 {len(anchor_names)} 个，回退到 variance 模式")
            anchor_mode = "variance"
        else:
            mode_desc = f"fixed-{len(anchor_names)} physically-motivated zone3 cols"
    if anchor_mode == "variance":
        z3_idx = [col2idx[c] for c in zone3_cols]
        X_z3   = X_norm_full[:, z3_idx]
        var    = X_z3.var(axis=0)
        sel    = np.argsort(-var)[:min(n_anchor, len(zone3_cols))]
        anchor_names = [zone3_cols[i] for i in sel]
        mode_desc = (f"top-{len(anchor_names)} variance (normal-train split only, "
                     "no cal/test labels used)")

    anchor_idx  = [col2idx[c] for c in anchor_names]
    effective_n = len(anchor_names)

    print(f"[WADI fair] zone3 共 {len(zone3_cols)} 列，"
          f"anchor_mode={anchor_mode}, 选取 {effective_n} 个 anchor")
    print(f"  mode_desc: {mode_desc}")
    print(f"  anchor: {anchor_names}")

                                                                       
    if client_zones is None:
        client_zones = ["zone1", "zone2"]

    clients = []
    all_anchor_names = anchor_names

    for zone_name in client_zones:
        own_cols = zone_map.get(zone_name, [])
        if not own_cols:
            print(f"  [WADI] zone={zone_name} 无有效列，跳过")
            continue

                        
        extra = []
        if include_global_in_anchor:
            extra = [c for c in zone_map.get("global", []) if c not in set(own_cols)]

        own_idx    = [col2idx[c] for c in own_cols]
        extra_idx  = [col2idx[c] for c in extra]
        client_idx = anchor_idx + own_idx + extra_idx
        n_k        = len(client_idx)
        feat_names = anchor_names + own_cols + extra

        split = int(round(len(X_norm_full) * (1.0 - cal_normal_frac)))
        split = min(max(split, window_len), len(X_norm_full) - window_len)
        X_train_raw = X_norm_full[:split, client_idx, None]
        X_cal_raw   = X_norm_full[split:, client_idx, None]
        X_test_raw  = X_atk_full[:,  client_idx, None]

        X_train, _ = _windowize(
            X_train_raw, np.zeros(len(X_train_raw), np.int64),
            window_len, stride, label_mode)
        X_cal, y_cal = _windowize(
            X_cal_raw, np.zeros(len(X_cal_raw), np.int64),
            window_len, stride, label_mode)
        X_test, y_test = _windowize(
            X_test_raw, y_atk_full, window_len, stride, label_mode)

        if max_train_windows and len(X_train) > max_train_windows:
            X_train = X_train[:max_train_windows]

        mem_mb = X_train.nbytes / 1e6
        if mem_mb > mem_warn_mb:
            print(f"  [WADI/{zone_name}] train split is about {mem_mb:.0f} MB")

        client = {
            "client_name":   zone_name,
            "feature_names": feat_names,
            "anchor_names":  anchor_names,
            "aux_names":     own_cols + extra,
            "anchor_zone":   "zone3",
            "X_train": X_train,
            "y_train": np.zeros(len(X_train), np.int64),
            "X_cal":   X_cal,
            "y_cal":   y_cal,
            "X_test":  X_test,
            "y_test":  y_test,
            "n_k":     n_k,
            "cal_anomaly_ratio": float(y_cal.mean()) if len(y_cal) else 0.0,
            "total_test_anom":   int(y_test.sum()),
            "is_sparse_anom":    int(y_test.sum()) < min_cal_anom,
            "split_protocol":    "paper_chronological_label_free",
        }
        clients.append(normalize_client_data(client))

        print(f"  {zone_name}: n_k={n_k} "
              f"(zone3_anchor={effective_n}+own={len(own_cols)}+global={len(extra)}), "
              f"train={len(X_train)}, "
              f"cal={len(X_cal)}(label-free), "
              f"test={len(X_test)}({int(y_test.sum())} anom, natural)")
        continue

        X_nrm_c = X_norm_full[:, client_idx, None]
        X_atk_c = X_atk_full[:,  client_idx, None]

        X_nw, _       = _windowize(X_nrm_c, y_norm_full, window_len, stride, label_mode)
        X_aw, y_aw    = _windowize(X_atk_c, y_atk_full,  window_len, stride, label_mode)

                 
        n_norm  = len(X_nw)
        n_train = max(256, int(n_norm * (1.0 - cal_normal_frac)))
        n_train = (min(n_train, n_norm - 128) if n_norm > 512
                   else max(1, int(n_norm * 0.8)))
        X_train    = X_nw[:n_train]
        X_hold     = X_nw[n_train:]

              
        mem_mb = X_train.nbytes / 1e6
        if mem_mb > mem_warn_mb:
            print(f"  ⚠ [WADI/{zone_name}] 训练集约 {mem_mb:.0f} MB")

                
        if max_train_windows and len(X_train) > max_train_windows:
            idx = rng.choice(len(X_train), max_train_windows, replace=False)
            idx.sort()
            X_train = X_train[idx]

        if len(X_hold) < 32:
            print(f"  [WADI/{zone_name}] 保留正常不足，跳过")
            continue

        n_cal_nrm = max(16, min(int(len(X_hold) * 0.4), len(X_hold) - 8))
        X_cal_nrm  = X_hold[:n_cal_nrm]
        X_test_nrm = X_hold[n_cal_nrm:]

        anom_mask  = y_aw == 1
        X_anom_p   = X_aw[anom_mask].copy()
        total_anom = len(X_anom_p)

        if total_anom < min_cal_anom:
            X_cal_anom  = np.empty((0,) + X_train.shape[1:], dtype=np.float32)
            X_test_anom = X_anom_p
        else:
            rng.shuffle(X_anom_p)
            n_by_r = max(0, int(round(
                len(X_cal_nrm) * cal_anom_ratio / max(1e-8, 1 - cal_anom_ratio))))
            n_cal_anom = max(min_cal_anom, n_by_r)
            n_cal_anom = min(n_cal_anom, total_anom - max(8, total_anom // 4))
            n_cal_anom = max(min_cal_anom, n_cal_anom)
            X_cal_anom    = X_anom_p[:n_cal_anom]
            X_test_anom_r = X_anom_p[n_cal_anom:]
            if test_anom_ratio and len(X_test_anom_r) > 0 and len(X_test_nrm) > 0:
                cap = int(len(X_test_nrm) * test_anom_ratio
                          / max(1e-8, 1 - test_anom_ratio))
                X_test_anom = X_test_anom_r[:max(8, cap)]
            else:
                X_test_anom = X_test_anom_r

        X_cal  = np.concatenate([X_cal_nrm, X_cal_anom])
        y_cal  = np.concatenate([np.zeros(len(X_cal_nrm), np.int64),
                                  np.ones(len(X_cal_anom),  np.int64)])
        p = rng.permutation(len(X_cal));  X_cal, y_cal = X_cal[p], y_cal[p]

        X_test = np.concatenate([X_test_nrm, X_test_anom])
        y_test = np.concatenate([np.zeros(len(X_test_nrm), np.int64),
                                  np.ones(len(X_test_anom), np.int64)])
        p = rng.permutation(len(X_test)); X_test, y_test = X_test[p], y_test[p]

        client = {
            "client_name":   zone_name,
            "feature_names": feat_names,
            "anchor_names":  anchor_names,
            "aux_names":     own_cols + extra,
            "anchor_zone":   "zone3",                            
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

        print(f"  {zone_name}: n_k={n_k} "
              f"(zone3_anchor={effective_n}+own={len(own_cols)}+global={len(extra)}), "
              f"train={len(X_train)}, "
              f"cal={len(X_cal)}({int(y_cal.sum())} anom), "
              f"test={len(X_test)}({int(y_test.sum())} anom)")

    if not clients:
        raise RuntimeError("WADI fair: 未能构建任何客户端，请检查数据文件和列名。")

    validate_federation_clients(clients, effective_n)
    return clients, effective_n, all_anchor_names
