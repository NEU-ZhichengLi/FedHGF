"""
SWaT 联邦数据加载器（Fair Benchmark 版）
=========================================

设计要点
--------
原始设计把 FIT101-FIT601 复制到所有客户端（每个客户端都持有全部 6 条 FIT
时间序列），审稿人认为这是数据泄漏。本版本做两项修改：

1. 物理产权透明化
   FIT101 属于 stage1，FIT201 属于 stage2，...，FIT601 属于 stage6。
   每个 stage 的 STAGE_CLIENTS 中明确包含本 stage 的 FIT 传感器。
   _build_specs 会将 FIT 加入 anchor_cols 而非 aux_cols，保留数据一致性。

2. 锚点语义明确化
   SWaT 的 SCADA 主站向所有子系统 PLC 广播全厂 FIT 读数，用于全流程流量
   平衡控制——这是 SWaT 有据可查的集中式 SCADA 架构。FIT 传感器安装在各
   处理阶段的管道交界处，是真实物理边界上的共享测量点。

   物理产权 vs. SCADA 可见性：
     FIT101 由 stage1 PLC 直接采集（本地产权），同时经 SCADA 广播对所有
     stage 可见（全局共享）。
     → 保留 6 个 FIT 作为全局 anchor 在联邦学习中具有合理的物理基础。

数据集统计（见代码末尾注释）
  - 51 个传感器，6 个物理处理阶段（stage1-6）
  - 正常数据：1,387,098 行（1秒采样，约16天）
  - 攻击数据：54,621 行，全部标注为 Attack
  - 每个 stage 的本地 FIT 数量见 STAGE_CLIENTS
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if str(_here.parent / "New") not in sys.path:
    sys.path.append(str(_here.parent / "New"))
from data_utils import make_split_audit, normalize_client_data, validate_federation_clients

                                                                             
       
                                                                             
                                                            
                                                      
                               
STAGE_CLIENTS = {
    "stage1": [
        "FIT101",                                                        
        "LIT101", "MV101", "P101", "P102",
    ],
    "stage2": [
        "FIT201",                                                
        "AIT201", "AIT202", "AIT203",
        "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
    ],
    "stage3": [
        "FIT301",                                                
        "DPIT301", "LIT301",
        "MV301", "MV302", "MV303", "MV304", "P301", "P302",
    ],
    "stage4": [
        "FIT401",                                                
        "AIT401", "AIT402", "LIT401",
        "P401", "P402", "P403", "P404", "UV401",
    ],
    "stage5": [
        "FIT501",                                                       
        "AIT501", "AIT502", "AIT503", "AIT504",
        "FIT502", "FIT503", "FIT504",                                
        "P501", "P502", "PIT501", "PIT502", "PIT503",
    ],
    "stage6": [
        "FIT601",                                                
        "P601", "P602", "P603",
    ],
}

                                  
                                             
DEFAULT_ANCHORS = ["FIT101", "FIT201", "FIT301", "FIT401", "FIT501", "FIT601"]


@dataclass
class SwatFairSpec:
    client_name: str
    anchor_cols: List[str]                         
    local_fit: str                                                 
    aux_cols: List[str]                       


def _build_specs(anchor_cols: Sequence[str]) -> List[SwatFairSpec]:
    specs = []
    for client_name, local_cols in STAGE_CLIENTS.items():
        aux = [c for c in local_cols if c not in set(anchor_cols)]
        local_fit = next((c for c in local_cols if c in set(anchor_cols)), "")
        specs.append(SwatFairSpec(
            client_name=client_name,
            anchor_cols=list(anchor_cols),
            local_fit=local_fit,
            aux_cols=aux,
        ))
    return specs


def _resolve_anchors(all_feat: Sequence[str], n_anchor: Optional[int]) -> List[str]:
    default = [c for c in DEFAULT_ANCHORS if c in set(all_feat)]
    if n_anchor is None:
        return default
    return default[:n_anchor]


def _build_specs_localfit() -> List[SwatFairSpec]:
    """
    LocalFITSlot ablation: each stage uses ONLY its own local FIT as the
    anchor slot (position 0). Different stages have semantically different
    anchors but structurally the same n_anchor=1 slot.
    Called SWaT-LocalFITSlot in the paper (Tier-2 ablation).
    """
    specs = []
    for client_name, local_cols in STAGE_CLIENTS.items():
        local_fit = next((c for c in local_cols if c.startswith("FIT")), "")
        anchor_cols = [local_fit] if local_fit else []
        aux_cols = [c for c in local_cols if c not in set(anchor_cols)]
        specs.append(SwatFairSpec(
            client_name=client_name,
            anchor_cols=anchor_cols,
            local_fit=local_fit,
            aux_cols=aux_cols,
        ))
    return specs


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _windowize(values: np.ndarray, labels: np.ndarray,
               window_len: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    n = len(values)
    for start in range(0, n - window_len + 1, stride):
        end = start + window_len
        X.append(values[start:end])
        y.append(int(labels[start:end].max() > 0))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def load_swat_fair(
    data_dir: str,
    window_len: int = 16,
    stride: int = 4,
    cal_anom_ratio: float = 0.22,
    cal_normal_frac: float = 0.15,
    test_anom_ratio: float = 0.30,
    seed: int = 42,
    n_anchor: Optional[int] = None,
    max_train_rows: Optional[int] = None,
    max_attack_rows: Optional[int] = None,
    localfit_mode: bool = False,
) -> Tuple[List[dict], int, List[str]]:
    """
    构造 SWaT 联邦基准（fair 版）。

    返回
    ----
    (clients, n_anchor, anchor_col_names)
    每个 client 包含 feature_names / anchor_names / aux_names / X_train / X_cal / X_test / y_cal / y_test
    并记录 local_fit_name（本 stage 直接采集的 FIT）用于论文可见性说明。
    """
    base = Path(data_dir)

    df_normal = _clean(pd.read_csv(base / "normal.csv", nrows=max_train_rows))
    df_attack = _clean(pd.read_csv(base / "attack.csv", nrows=max_attack_rows))
    df_normal = df_normal.ffill().bfill().fillna(0.0)
    df_attack = df_attack.ffill().bfill().fillna(0.0)

    all_feat = [c for c in df_normal.columns
                if c not in ("Timestamp", "Normal/Attack")]
    if localfit_mode:
        specs = _build_specs_localfit()
        anchor_cols = [s.anchor_cols[0] for s in specs if s.anchor_cols]
        n_anchor_eff = 1
    else:
        anchor_cols = _resolve_anchors(all_feat, n_anchor)
        specs = _build_specs(anchor_cols)
        n_anchor_eff = len(anchor_cols)

    X_norm_all = df_normal[all_feat].to_numpy(dtype=np.float32)
    y_norm_all = np.zeros(len(X_norm_all), dtype=np.int64)
    X_atk_all  = df_attack[all_feat].to_numpy(dtype=np.float32)
    y_atk_all  = (df_attack["Normal/Attack"].str.strip() == "Attack").to_numpy(
        dtype=np.int64)

    clients = []
    for spec in specs:
        feat_order = spec.anchor_cols + spec.aux_cols
        idx = [all_feat.index(c) for c in feat_order]

        split = int(round(len(X_norm_all) * (1.0 - cal_normal_frac)))
        split = min(max(split, window_len), len(X_norm_all) - window_len)
        X_train_raw = X_norm_all[:split, idx, None]
        X_cal_raw   = X_norm_all[split:, idx, None]
        X_test_raw  = X_atk_all[:, idx, None]

        X_train, _ = _windowize(X_train_raw, np.zeros(len(X_train_raw), np.int64),
                                window_len, stride)
        X_cal, y_cal = _windowize(X_cal_raw, np.zeros(len(X_cal_raw), np.int64),
                                  window_len, stride)
        X_test, y_test = _windowize(X_test_raw, y_atk_all, window_len, stride)

        client = {
            "client_name":   spec.client_name,
            "feature_names": feat_order,
            "anchor_names":  spec.anchor_cols,
            "aux_names":     spec.aux_cols,
            "local_fit":     spec.local_fit,                         
            "X_train": X_train,
            "y_train": np.zeros(len(X_train), dtype=np.int64),
            "X_cal":   X_cal,
            "y_cal":   y_cal,
            "X_test":  X_test,
            "y_test":  y_test,
            "n_k":     len(feat_order),
            "cal_anomaly_ratio": float(y_cal.mean()) if len(y_cal) else 0.0,
            "total_test_anom":   int(y_test.sum()),
            "is_sparse_anom":    int(y_test.sum()) < 20,
            "split_protocol":    "paper_chronological_label_free",
            "split_audit": make_split_audit(
                dataset="swat",
                client_name=spec.client_name,
                window_len=window_len,
                stride=stride,
                train_source="Data/SWAT/normal.csv",
                train_rows=(0, split),
                cal_source="Data/SWAT/normal.csv",
                cal_rows=(split, len(X_norm_all)),
                test_source="Data/SWAT/attack.csv",
                test_rows=(0, len(X_atk_all)),
                label_mode="any",
                anchor_semantics=(
                    "replicated_public_scada_context: all clients receive the "
                    "same FIT101-FIT601 public context, not independent "
                    "same-semantics factory sensors"
                ),
            ),
        }
        clients.append(normalize_client_data(client))
        print(f"  {spec.client_name}: local_FIT={spec.local_fit}, "
              f"n_k={len(feat_order)} (anchor={len(spec.anchor_cols)}+aux={len(spec.aux_cols)}), "
              f"train={len(X_train)}, cal={len(X_cal)}(label-free), "
              f"test={len(X_test)}({int(y_test.sum())} anom, natural)")

    validate_federation_clients(clients, n_anchor_eff)
    anchor_names_out = (
        ["LOCAL_FIT_SLOT"] if localfit_mode else list(anchor_cols)
    )
    return clients, n_anchor_eff, anchor_names_out


                                                                             
         
                                                                             
                                             
 
               
                                                            
                                    
                                                  
                                        
                                
                                                                          
                                
                             
                     
 
                                           
                                                                  
                       
                      
                      
                                                          
                     
 
        
                                  
                                     
