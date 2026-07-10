"""
BATADAL 联邦数据加载器（Fair Benchmark 版 — L_T 水箱液位锚点设计）
================================================================

数据集统计
----------
  BATADAL（Battle of the Attack Detection Algorithms）
  Water Distribution Network with 9 reservoirs/tanks + pump stations

  训练集：BATADAL_dataset03.csv  8,761 行（1小时采样），全部正常，ATT_FLAG=0
  验证集：BATADAL_dataset04.csv  4,177 行，含219行攻击（ATT_FLAG=1），其余-999=未标注
  测试集：BATADAL_test_dataset.csv（可选）

  总特征数：43 个传感器
    L_T1-7  : 水箱液位（7个）
    F_PU1-11, F_V2 : 泵/阀流量（12个）
    S_PU1-11, S_V2 : 泵/阀开关状态（12个）
    P_J*          : 管网结点压力（12个）

锚点设计：L_T（水箱液位）
-------------------------
水箱液位是水分配网络的"全局状态变量"：
  - 全网供需平衡直接反映在液位变化上
  - 任意一处泵故障或阀门攻击都会通过水量变化影响液位
  - SCADA 主站向所有泵站控制器广播液位读数，用于自动调度

7 个水箱（T1-T7）覆盖整个配水网络，液位传感器 L_T1-7 是最自然的
"跨子系统公共观测量"，在真实 BATADAL 系统中对所有控制节点可见。

联邦划分
--------
客户端按泵站功能区划分：
  ZoneA（主泵站）: PU1-PU6 — 供应 T2/T3/T4/T6
  ZoneB（次泵站）: PU7-PU11 + V2 + 所有结点压力 — 供应 T5/T7 及末端

每个客户端特征：
  ZoneA: [L_T1-7（7个锚点）] + [F_PU1-6, S_PU1-6（12个aux）] = 19维
  ZoneB: [L_T1-7（7个锚点）] + [F_PU7-11, F_V2, S_PU7-11, S_V2, P_J*（24个aux）] = 31维
  异构度：(31-19)/31 = 38.7%（与SWaT相当的合理异构度）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if str(_here.parent / "New") not in sys.path:
    sys.path.append(str(_here.parent / "New"))
from data_utils import normalize_client_data, validate_federation_clients

                                                                             
       
                                                                             
                              
ANCHOR_COLS = ["L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7"]

             
ZONE_A_AUX = [
    "F_PU1", "S_PU1",
    "F_PU2", "S_PU2",
    "F_PU3", "S_PU3",
    "F_PU4", "S_PU4",
    "F_PU5", "S_PU5",
    "F_PU6", "S_PU6",
]

ZONE_B_AUX = [
    "F_PU7",  "S_PU7",
    "F_PU8",  "S_PU8",
    "F_PU9",  "S_PU9",
    "F_PU10", "S_PU10",
    "F_PU11", "S_PU11",
    "F_V2",   "S_V2",
    "P_J280", "P_J269", "P_J300", "P_J256",
    "P_J289", "P_J415", "P_J302", "P_J306",
    "P_J307", "P_J317", "P_J14",  "P_J422",
]

CLIENT_ZONES = {
    "zoneA": ZONE_A_AUX,                 
    "zoneB": ZONE_B_AUX,                                
}

META_COLS = {"DATETIME", "ATT_FLAG"}


def _windowize(X: np.ndarray, y: np.ndarray,
               wlen: int, stride: int,
               label_mode: str = "any") -> Tuple[np.ndarray, np.ndarray]:
    wins, labs = [], []
    for s in range(0, len(X) - wlen + 1, stride):
        wins.append(X[s: s + wlen])
        chunk = y[s: s + wlen]
        if label_mode == "last":
            labs.append(int(chunk[-1] > 0))
        elif label_mode == "center":
            labs.append(int(chunk[wlen // 2] > 0))
        elif label_mode == "majority":
            labs.append(int(chunk.sum() > wlen // 2))
        else:         
            labs.append(int(chunk.max() > 0))
    if not wins:
        return np.empty((0, wlen) + X.shape[1:], np.float32), np.empty(0, np.int64)
    return np.asarray(wins, np.float32), np.asarray(labs, np.int64)


def load_batadal_fair(
    data_dir: str,
    window_len: int = 16,
    stride: int = 4,
    cal_anom_ratio: float = 0.10,
    cal_normal_frac: float = 0.15,
    test_anom_ratio: float = 0.15,
    min_cal_anom: int = 16,
    seed: int = 42,
    max_train_rows: Optional[int] = None,
    label_mode: str = "any",
) -> Tuple[List[dict], int, List[str]]:
    """
    构造 BATADAL 联邦基准（fair 版）。

    数据策略
    --------
    - 训练集：dataset03（全正常）的前 80%
    - 校准集：dataset03 后 20% + dataset04 标注攻击部分
    - 测试集：dataset04 全部（ATT_FLAG=1 为攻击，ATT_FLAG=-999 视为正常）
    """
    rng = np.random.RandomState(seed)
    base = Path(data_dir)

                                                                         
    df3 = pd.read_csv(base / "BATADAL_dataset03.csv", nrows=max_train_rows)
    df3.columns = [c.strip() for c in df3.columns]
    df3 = df3.ffill().bfill().fillna(0.0)

    df4 = pd.read_csv(base / "BATADAL_dataset04.csv")
    df4.columns = [c.strip() for c in df4.columns]
    df4 = df4.ffill().bfill().fillna(0.0)

    all_feat = [c for c in df3.columns if c not in META_COLS]

                       
    missing = [c for c in ANCHOR_COLS if c not in all_feat]
    if missing:
        raise ValueError(f"BATADAL: 缺少 anchor 列 {missing}。请确认数据格式。")

    X_train_all = df3[all_feat].to_numpy(np.float32)
    y_train_all = np.zeros(len(X_train_all), np.int64)                  

    X_val_all   = df4[all_feat].to_numpy(np.float32)
    y_val_all   = (df4["ATT_FLAG"] == 1).to_numpy(np.int64)                      

    print(f"[BATADAL fair] train={len(X_train_all)}, val={len(X_val_all)} "
          f"(attack={y_val_all.sum()})")
    print(f"  anchor: {ANCHOR_COLS} (n={len(ANCHOR_COLS)})")

    clients = []
    for zone_name, aux_cols in CLIENT_ZONES.items():
        missing_aux = [c for c in aux_cols if c not in all_feat]
        if missing_aux:
            print(f"  ⚠ {zone_name}: 缺少 {len(missing_aux)} 个aux列（跳过缺失列）")
            aux_cols = [c for c in aux_cols if c in all_feat]

        feat_order = ANCHOR_COLS + aux_cols
        idx = [all_feat.index(c) for c in feat_order]
        n_k = len(feat_order)

        X_tr = X_train_all[:, idx, None]                
        X_vl = X_val_all[:,   idx, None]

        split = int(round(len(X_tr) * (1.0 - cal_normal_frac)))
        split = min(max(split, window_len), len(X_tr) - window_len)
        X_train, _ = _windowize(
            X_tr[:split], np.zeros(split, np.int64),
            window_len, stride, label_mode)
        X_cal, y_cal = _windowize(
            X_tr[split:], np.zeros(len(X_tr) - split, np.int64),
            window_len, stride, label_mode)
        X_test, y_test = _windowize(X_vl, y_val_all, window_len, stride, label_mode)

        client = {
            "client_name":   zone_name,
            "feature_names": feat_order,
            "anchor_names":  ANCHOR_COLS,
            "aux_names":     aux_cols,
            "anchor_type":   "tank_level",
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
              f"(L_T_anchor={len(ANCHOR_COLS)}+aux={len(aux_cols)}), "
              f"train={len(X_train)}, "
              f"cal={len(X_cal)}(label-free), "
              f"test={len(X_test)}({int(y_test.sum())} anom, natural)")
        continue

        X_tr_w, _       = _windowize(X_tr, y_train_all, window_len, stride, label_mode)
        X_vl_w, y_vl_w  = _windowize(X_vl, y_val_all,  window_len, stride, label_mode)
        if label_mode != "any":
            print(f"    [{zone_name}] label_mode={label_mode}, "
                  f"val anom windows: {int(y_vl_w.sum())} / {len(y_vl_w)}")

                   
        n_tr = len(X_tr_w)
        n_train = max(64, int(n_tr * (1.0 - cal_normal_frac)))
        n_train = min(n_train, n_tr - 32)
        X_train    = X_tr_w[:n_train]
        X_cal_nrm  = X_tr_w[n_train:]

                            
        anom_mask  = y_vl_w == 1
        X_anom     = X_vl_w[anom_mask].copy()
        X_val_nrm  = X_vl_w[~anom_mask].copy()
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

                                                               
        rng.shuffle(X_val_nrm)
        if test_anom_ratio and len(X_test_anom) > 0 and len(X_val_nrm) > 0:
            n_test_norm_target = int(round(
                len(X_test_anom) * (1.0 - test_anom_ratio)
                / max(1e-8, test_anom_ratio)
            ))
            n_test_norm = min(len(X_val_nrm), max(32, n_test_norm_target))
            X_test_nrm_final = X_val_nrm[:n_test_norm]
        else:
            X_test_nrm_final = X_val_nrm

        X_cal  = np.concatenate([X_cal_nrm, X_cal_anom])
        y_cal  = np.concatenate([np.zeros(len(X_cal_nrm), np.int64),
                                  np.ones(len(X_cal_anom),  np.int64)])
        p = rng.permutation(len(X_cal)); X_cal, y_cal = X_cal[p], y_cal[p]

        X_test = np.concatenate([X_test_nrm_final, X_test_anom])
        y_test = np.concatenate([np.zeros(len(X_test_nrm_final), np.int64),
                                  np.ones(len(X_test_anom),       np.int64)])
        p = rng.permutation(len(X_test)); X_test, y_test = X_test[p], y_test[p]

        client = {
            "client_name":   zone_name,
            "feature_names": feat_order,
            "anchor_names":  ANCHOR_COLS,
            "aux_names":     aux_cols,
            "anchor_type":   "tank_level",                
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
              f"(L_T_anchor={len(ANCHOR_COLS)}+aux={len(aux_cols)}), "
              f"train={len(X_train)}, "
              f"cal={len(X_cal)}({int(y_cal.sum())} anom), "
              f"test={len(X_test)}({int(y_test.sum())} anom)")

    validate_federation_clients(clients, len(ANCHOR_COLS))
    return clients, len(ANCHOR_COLS), list(ANCHOR_COLS)
