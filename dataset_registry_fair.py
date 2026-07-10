"""
Dataset registry for the heterogeneous federated benchmark.

Tier 1 datasets use physical or public-state anchors.
Tier 2 datasets use public-context anchors.
Tier 3 datasets are auxiliary functional-alignment settings.

Each registry entry defines the data loader, client partition, anchor protocol,
default model configuration, and output directory used by the experiment runners.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if str(_here.parent / "New") not in sys.path:
    sys.path.append(str(_here.parent / "New"))

from data_loader_swat_fair    import load_swat_fair
from data_loader_wadi_fair    import load_wadi_fair
from data_loader_batadal_fair import load_batadal_fair
from data_loader_smd_fair     import load_smd_fair
from data_loader_smd_grouped  import load_smd_grouped
from data_loader_hai_fed      import load_hai_federation


                                                               
BASE_CFG_FAIR = dict(
    n_rounds=10,
    local_epochs=1,
    d_h=64,
    flow_blocks=4,
    flow_hidden=128,
                                                                       
                                                            
                                
    batch_size=128,
    lr=3e-4,
    flow_lr=3e-4,
    lambda_c=1.0,
    lambda_g=0.005,                                                             
    lambda_t=0.005,                                                             
    lambda_v=0.2,                                                                      
    gamma_var=0.05,                                               
    C_g=1.0,
    sigma_g=1.0,
    C_theta=1.0,
    sigma_theta=1.0,
    C_c=1.0,
    sigma_c=1.0,
    use_dp=True,
    adaptive_threshold_mode="quantile",
    target_anom_rate=0.15,                                                              
    use_calibration=True,
    use_prediction_loss=False,                                                                        
    center_score_mode="local",
    use_label_assisted_fusion=False,                                                   
    score_mode="both",
    use_flow=True,
    collapse_std_thr=0.001,
    hybrid_center_alpha=0.0,                                                          
    adaptive_hybrid_alpha=False,
    w_fusion=(0.20, 0.55, 0.25),                                                 
    use_data_driven_cross_block=True,                                                 
    graph_residual_mode="node_topk_q90",                                
    relation_value_weight=0.5,
                                               
                                                              
                                                                      
    encoder_type="npformer_gp",
    patch_len=4,                                                                
    patch_stride=2,
    tf_layers=2,
    tf_heads=4,
    tf_ffn=128,
    tf_dropout=0.1,
                                                                                 
                                                               
                                                                               
)

                                                               
BASELINE_CATEGORIES = {
    "federated": ["fl_stam", "fedanomaly", "pefad"],
    "local_client": ["gdn", "ganf", "mtad_gat", "tranad"],
}

DATASET_REGISTRY_FAIR: Dict[str, Dict[str, Any]] = {

                                                                            
                                           
                                                                            

                                                             
    "wadi": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor",
        "loader": load_wadi_fair,
        "default_data_dir": str(Path("Data") / "WADI"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            cal_anom_ratio=0.10, test_anom_ratio=0.15, cal_normal_frac=0.15,
            anchor_mode="all15",                                       
            min_cal_anom=50,
            max_train_rows=120_000,
            include_global_in_anchor=True,
            client_zones=["zone1", "zone2"],
        ),
        "cfg_overrides": dict(
            n_rounds=12, local_epochs=1, d_h=64, flow_hidden=128,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1] WADI zone3 全部 15 变量作 anchor (all15). "
            "zone3 是 zone1/zone2 的公共物理下游储水系统，所有 15 个传感器均有物理依据。"
            "主实验使用 all15 以避免 anchor 选择偏差；fixed8 仅作消融对比。"
        ),
        "out_dir": "results_wadi_fair",
    },

                                                                    
    "wadi_all15": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (ablation: all-15)",
        "loader": load_wadi_fair,
        "default_data_dir": str(Path("Data") / "WADI"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            cal_anom_ratio=0.10, test_anom_ratio=0.15, cal_normal_frac=0.15,
            anchor_mode="all15",                              
            min_cal_anom=50,
            max_train_rows=120_000,
            include_global_in_anchor=True,
            client_zones=["zone1", "zone2"],
        ),
        "cfg_overrides": dict(
            n_rounds=12, local_epochs=1, d_h=64, flow_hidden=128,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 ablation] WADI zone3 all-15 anchor. "
            "全部 15 个 zone3 传感器作 anchor，无选择偏差，用于与 fixed8 对比。"
        ),
        "out_dir": "results_wadi_all15",
    },

                                                             
    "wadi_variance": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (ablation: variance)",
        "loader": load_wadi_fair,
        "default_data_dir": str(Path("Data") / "WADI"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            cal_anom_ratio=0.10, test_anom_ratio=0.15, cal_normal_frac=0.15,
            anchor_mode="variance",                          
            n_anchor=8,
            min_cal_anom=50,
            max_train_rows=120_000,
            include_global_in_anchor=True,
            client_zones=["zone1", "zone2"],
        ),
        "cfg_overrides": dict(
            n_rounds=12, local_epochs=1, d_h=64, flow_hidden=128,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 ablation] WADI zone3 top-8 variance anchor. "
            "Anchor selection is performed only on the normal training split of zone3, "
            "without using calibration or test labels."
        ),
        "out_dir": "results_wadi_variance",
    },

                                                                     
    "batadal": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor",
        "loader": load_batadal_fair,
        "default_data_dir": str(Path("Data") / "BATADAL"),
        "loader_kwargs": dict(
            window_len=16, stride=1,
            cal_anom_ratio=0.10, cal_normal_frac=0.15, test_anom_ratio=0.15,
            min_cal_anom=30,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1] BATADAL L_T1–L_T7 水箱液位 anchor. "
            "水箱液位是配水网络全局状态变量，所有泵站控制器通过 SCADA 读取液位进行调度。"
            "ZoneA: PU1-6 (12 aux), ZoneB: PU7-11+V2+P_J* (24 aux)."
        ),
        "out_dir": "results_batadal_fair",
    },

                                                             
                                                                 
                                                                  
                                                       
    "batadal_more_rounds": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (more-rounds variant)",
        "loader": load_batadal_fair,
        "default_data_dir": str(Path("Data") / "BATADAL"),
        "loader_kwargs": dict(
            window_len=16, stride=1,
            cal_anom_ratio=0.10, cal_normal_frac=0.15, test_anom_ratio=0.15,
            min_cal_anom=30,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=20, local_epochs=1,                               
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / BATADAL more-rounds] 同 batadal，唯一差异: n_rounds 10→20 "
            "(验证 NPFormer-GP 在 BATADAL 的 s1 弱化是否由欠训练导致)."
        ),
        "out_dir": "results_batadal_more_rounds_fair",
    },

                                                               
    "hai": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor",
        "loader": load_hai_federation,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            processes=["P1", "P2", "P4"],
            window_len=16, stride=4,
            cal_anom_ratio=0.10, cal_normal_frac=0.08, test_anom_ratio=0.15,
            max_train_rows=200_000, min_cal_anom=100,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1] HAI P3 水筒 anchor (7 vars): "
            "P3_FIT01, P3_LCP01D, P3_LCV01D, P3_LH, P3_LIT01, P3_LL, P3_PIT01. "
            "P3 是 P1/P2 的公共给水来源，物理真实共享。"
        ),
        "out_dir": "results_hai_fair",
    },

                                                              
                                                          
                                                               
                                                              
                                           
    "hai_lowcenter": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (low-center variant)",
        "loader": load_hai_federation,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            processes=["P1", "P2"],
            window_len=16, stride=4,
            cal_anom_ratio=0.10, cal_normal_frac=0.08, test_anom_ratio=0.15,
            max_train_rows=200_000, min_cal_anom=100,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            lambda_c=0.01,                                               
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / HAI low-center] 同 hai，唯一差异: lambda_c 0.02→0.01 "
            "(降低 center alignment 压力, 防止 NPFormer-GP representation collapse)."
        ),
        "out_dir": "results_hai_lowcenter_fair",
    },

                                                              
                                                       
                                                                   
                                                             
    "batadal_light": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (lightweight transformer variant)",
        "loader": load_batadal_fair,
        "default_data_dir": str(Path("Data") / "BATADAL"),
        "loader_kwargs": dict(
            window_len=16, stride=1,                                   
            cal_anom_ratio=0.10, cal_normal_frac=0.15, test_anom_ratio=0.15,
            min_cal_anom=30,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
            tf_layers=1,                                            
            tf_ffn=64,                                          
            tf_dropout=0.2,                                      
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / BATADAL light] 同 batadal，差异: tf_layers 2→1, "
            "tf_ffn 128→64, dropout 0.1→0.2. 数据协议 (T/stride/patch) 不动. "
            "假设: 小样本低频场景下 2-layer patch transformer 过拟合."
        ),
        "out_dir": "results_batadal_light_fair",
    },

                                                                     
                                                                       
                                                                     
                                                      
                                                           
    "batadal_light_lc": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (light + low-center variant)",
        "loader": load_batadal_fair,
        "default_data_dir": str(Path("Data") / "BATADAL"),
        "loader_kwargs": dict(
            window_len=16, stride=1,
            cal_anom_ratio=0.10, cal_normal_frac=0.15, test_anom_ratio=0.15,
            min_cal_anom=30,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
            tf_layers=1, tf_ffn=64, tf_dropout=0.2,            
            lambda_c=0.01,                                              
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / BATADAL light+low-center] 同 batadal_light, "
            "唯一额外差异: lambda_c 0.02→0.01, 防止 light encoder 容量"
            "瘦身后被 center loss 拉塌."
        ),
        "out_dir": "results_batadal_light_lc_fair",
    },

                                                                        
                                                              
                                                  
    "hai_t32": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (mid-window variant)",
        "loader": load_hai_federation,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            processes=["P1", "P2"],
            window_len=32, stride=4,
            cal_anom_ratio=0.10, cal_normal_frac=0.08, test_anom_ratio=0.15,
            max_train_rows=200_000, min_cal_anom=100,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            patch_len=4, patch_stride=2,                                     
            lambda_c=0.01,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / HAI mid-window] 同 hai, 差异: window_len 16→32, "
            "patch=4/2 (P=15), lambda_c=0.01. 中间窗长."
        ),
        "out_dir": "results_hai_t32_fair",
    },

                                                                      
                                                                     
                                                                  
                                                                       
                                                    
                                                                             
    "hai_long64": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (long-window variant)",
        "loader": load_hai_federation,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            processes=["P1", "P2"],
            window_len=64, stride=8,                                              
            cal_anom_ratio=0.10, cal_normal_frac=0.08, test_anom_ratio=0.15,
            max_train_rows=200_000, min_cal_anom=100,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            patch_len=8, patch_stride=4,                                             
            lambda_c=0.01,                                                        
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / HAI long-window] 同 hai，差异: window_len 16→64 "
            "(P=15 patch tokens, transformer 充分上下文), lambda_c 0.02→0.01 "
            "(给 variance-floor 留空间, 抑制 representation collapse)."
        ),
        "out_dir": "results_hai_long64_fair",
    },

                                                                 
                                                       
                                                                   
                          
    "batadal_small": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (lightweight transformer)",
        "loader": load_batadal_fair,
        "default_data_dir": str(Path("Data") / "BATADAL"),
        "loader_kwargs": dict(
            window_len=32, stride=1,                                      
            cal_anom_ratio=0.10, cal_normal_frac=0.15, test_anom_ratio=0.15,
            min_cal_anom=30,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.003,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / BATADAL small] 同 batadal，差异: window_len 16→32, "
            "patch=8/4 (P=7), tf_layers 2→1, tf_ffn 128→64, dropout 0.1→0.2, "
            "lambda_c 0.02→0.01. 假设: 小数据集低频场景下原 NPFormer-GP 过拟合."
        ),
        "out_dir": "results_batadal_small_fair",
    },

                                                              
                                                               
                                                                     
                                                  
                                                    
    "hai_anticollapse": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (anti-collapse variant)",
        "loader": load_hai_federation,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            processes=["P1", "P2"],
            window_len=16, stride=4,
            cal_anom_ratio=0.10, cal_normal_frac=0.08, test_anom_ratio=0.15,
            max_train_rows=200_000, min_cal_anom=100,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            lambda_v=0.4,                                              
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 1 / HAI anti-collapse] 同 hai，唯一差异: lambda_v 0.2→0.4 "
            "(2x variance-floor regularizer). 用于验证 HAI 上 NPFormer-GP 的 "
            "embed_std 衰减是否由 variance-floor 强度不足导致."
        ),
        "out_dir": "results_hai_anticollapse_fair",
    },

                                                                            
                                    
                                                                            

                                                                      
    "swat": {
        "tier": 2,
        "tier_desc": "Public-Context Anchor",
        "loader": load_swat_fair,
        "default_data_dir": str(Path("Data") / "SWAT"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            cal_anom_ratio=0.22, cal_normal_frac=0.15, test_anom_ratio=0.30,
            n_anchor=None,                 
        ),
        "cfg_overrides": dict(
            n_rounds=15, local_epochs=1, d_h=64, flow_hidden=128,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 2] SWaT FIT-public-context anchor. "
            "FIT101–FIT601 是全厂 SCADA 公共流量上下文，所有 stage PLC 可读取。"
            "注：每个 stage 对本地 FIT 有直接采集产权（local_fit 字段标注）；"
            "其他 FIT 通过 SCADA 广播可见。论文中标注 public-context setting。"
        ),
        "out_dir": "results_swat_fair",
    },

                                                                  
    "swat_localfit": {
        "tier": 2,
        "tier_desc": "Public-Context Anchor ablation: LocalFITOnly",
        "loader": load_swat_fair,
        "default_data_dir": str(Path("Data") / "SWAT"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            cal_anom_ratio=0.22, cal_normal_frac=0.15, test_anom_ratio=0.30,
            localfit_mode=True,                                          
        ),
        "cfg_overrides": dict(
            n_rounds=10, local_epochs=1, d_h=64, flow_hidden=128,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 2 ablation] SWaT LocalFITOnly: "
            "每个 stage 只使用本 stage 直接采集的 FIT（n_anchor=1），"
            "不使用 SCADA 广播的其他 stage FIT。"
            "用于验证全局 FIT anchor 是否真正提供信息增益。"
        ),
        "out_dir": "results_swat_localfit",
    },

                                                                            
                                                 
                                                                            

                                                                 
    "smd": {
        "tier": 3,
        "tier_desc": "Functional-Alignment Anchor (auxiliary experiment)",
        "loader": load_smd_fair,
        "default_data_dir": str(Path("Data") / "SMD"),
        "loader_kwargs": dict(
            n_anchor=8,
            window_len=20, stride=5,
            cal_normal_frac=0.15, cal_anom_ratio=0.10, test_anom_ratio=0.10,
            min_cal_anom=20,
            exclude_machines=["machine-1-3", "machine-1-4"],
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            use_label_assisted_fusion=False,
            target_anom_rate=0.10,                                         
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 3 / SMD-FunctionalAlignment] "
            "Anchor = 跨 28 台机器平均方差 top-8 全局固定 feature index。"
            "所有客户端使用相同的 anchor slot 含义（功能位置一致）。"
            "注：不声称物理共享语义，属功能对齐基准，放辅助表报告。"
        ),
        "out_dir": "results_smd_fair",
    },

                                                                 
    "smd_grouped": {
        "tier": 3,
        "tier_desc": "Grouped Federation (3 server clusters as clients)",
        "loader": load_smd_grouped,
        "default_data_dir": str(Path("Data") / "SMD"),
        "loader_kwargs": dict(
            n_anchor=8,
            window_len=20, stride=5,
            cal_normal_frac=0.15, cal_anom_ratio=0.10, test_anom_ratio=0.10,
            min_cal_anom=30,
            max_train_rows=200000,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            use_label_assisted_fusion=False,
            target_anom_rate=0.10,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 3 / SMD-Grouped] "
            "3 server clusters (machine-1/2/3) as 3 federated clients. "
            "Anchor = top-8 global mean-variance features (cross-cluster). "
            "All clients share identical 38-dim feature schema."
        ),
        "out_dir": "results_smd_grouped_fair",
    },

                                                                    
    "smd_all38": {
        "tier": 3,
        "tier_desc": "Homogeneous Federated Benchmark (all-38 shared features)",
        "loader": load_smd_fair,
        "default_data_dir": str(Path("Data") / "SMD"),
        "loader_kwargs": dict(
            n_anchor=38,
            window_len=20, stride=5,
            cal_normal_frac=0.15, cal_anom_ratio=0.10, test_anom_ratio=0.10,
            min_cal_anom=20,
            exclude_machines=["machine-1-3", "machine-1-4"],
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            use_label_assisted_fusion=False,
            target_anom_rate=0.10,
        ),
        "cfg_overrides_full": dict(use_label_assisted_fusion=True),
        "anchor_design": (
            "[Tier 3 / SMD-Homogeneous] "
            "所有 38 个 feature 均作为 anchor（n_anchor=38，无 aux）。"
            "SMD 26 台服务器特征 schema 完全一致（f0-f37）。"
            "此配置作为同质化联邦基准，anchor 为全维度共享。"
        ),
        "out_dir": "results_smd_all38_fair",
    },
}


                                                                          

def get_fair_spec(dataset: str) -> Dict[str, Any]:
    key = dataset.lower()
    if key not in DATASET_REGISTRY_FAIR:
        raise KeyError(
            f"Fair registry 不支持数据集: '{dataset}'。\n"
            f"  Tier-1 主实验: wadi, wadi_all15, wadi_variance, batadal, hai\n"
            f"  Tier-2 主实验: swat, swat_localfit\n"
            f"  Tier-3 辅助: smd"
        )
    return DATASET_REGISTRY_FAIR[key]


def list_datasets_by_tier() -> None:
    from collections import defaultdict
    by_tier: Dict[int, list] = defaultdict(list)
    for name, spec in DATASET_REGISTRY_FAIR.items():
        by_tier[spec["tier"]].append((name, spec["tier_desc"]))
    tier_labels = {1: "Tier 1 - Physical/Public-State",
                   2: "Tier 2 - Public-Context",
                   3: "Tier 3 - Functional-Alignment"}
    for t in sorted(by_tier.keys()):
        print(f"\n  {tier_labels[t]}")
        for name, desc in by_tier[t]:
            print(f"    {name:<22} {desc}")
