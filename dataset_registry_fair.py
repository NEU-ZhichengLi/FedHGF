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
from src.fedhgf.data.protocol_builder import (
    build_hai_shared_context_protocol,
    to_legacy_clients,
)


def load_hai_protocol_v2(
    data_dir: str,
    window_len: int = 16,
    stride: int = 4,
    train_fraction: float = 0.80,
    guard_gap: int = 15,
    max_train_rows: int | None = 200_000,
    label_mode: str = "any",
    seed: int | None = None,
):
    _ = seed
    federation = build_hai_shared_context_protocol(
        data_dir,
        window_length=window_len,
        stride=stride,
        train_fraction=train_fraction,
        guard_gap=guard_gap,
        label_mode=label_mode,
        max_train_rows=max_train_rows,
    )
    return to_legacy_clients(federation)


                                                               
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
    use_calibration=True,
    use_prediction_loss=False,                                                                        
    center_score_mode="local",
    use_label_assisted_fusion=False,                                                   
    score_mode="both",
    use_flow=True,
    collapse_std_thr=0.001,
    hybrid_center_alpha=0.0,                                                          
    adaptive_hybrid_alpha=False,
    w_fusion=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
        "out_dir": "results_batadal_more_rounds_fair",
    },

                                                               
    "hai": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor",
        "loader": load_hai_protocol_v2,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            train_fraction=0.80,
            guard_gap=15,
            max_train_rows=200_000,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
        ),
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
        "out_dir": "results_hai_fair",
    },

                                                              
                                                          
                                                               
                                                              
                                           
    "hai_lowcenter": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (low-center variant)",
        "loader": load_hai_protocol_v2,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            train_fraction=0.80,
            guard_gap=15,
            max_train_rows=200_000,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            lambda_c=0.01,                                               
        ),
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
        "out_dir": "results_batadal_light_lc_fair",
    },

                                                                        
                                                              
                                                  
    "hai_t32": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (mid-window variant)",
        "loader": load_hai_protocol_v2,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            window_len=32, stride=4,
            train_fraction=0.80,
            guard_gap=31,
            max_train_rows=200_000,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            patch_len=4, patch_stride=2,                                     
            lambda_c=0.01,
        ),
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
        "out_dir": "results_hai_t32_fair",
    },

                                                                      
                                                                     
                                                                  
                                                                       
                                                    
                                                                             
    "hai_long64": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (long-window variant)",
        "loader": load_hai_protocol_v2,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            window_len=64, stride=8,                                              
            train_fraction=0.80,
            guard_gap=63,
            max_train_rows=200_000,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            patch_len=8, patch_stride=4,                                             
            lambda_c=0.01,                                                        
        ),
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
        "out_dir": "results_batadal_small_fair",
    },

                                                              
                                                               
                                                                     
                                                  
                                                    
    "hai_anticollapse": {
        "tier": 1,
        "tier_desc": "Physical/Public-State Anchor (anti-collapse variant)",
        "loader": load_hai_protocol_v2,
        "default_data_dir": str(Path("Data") / "HAI 21.03"),
        "loader_kwargs": dict(
            window_len=16, stride=4,
            train_fraction=0.80,
            guard_gap=15,
            max_train_rows=200_000,
        ),
        "cfg_overrides": dict(
            d_h=64, n_rounds=10, local_epochs=1,
            collapse_std_thr=0.001, flow_hidden=128,
            use_label_assisted_fusion=False,
            lambda_v=0.4,                                              
        ),
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
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
        "cfg_overrides_full": dict(),
        "anchor_design": "Defined by the canonical protocol manifest.",
        "out_dir": "results_smd_all38_fair",
    },
}


                                                                          

def get_fair_spec(dataset: str) -> Dict[str, Any]:
    key = dataset.lower()
    if key not in DATASET_REGISTRY_FAIR:
        raise KeyError(
            f"Fair registry 涓嶆敮鎸佹暟鎹泦: '{dataset}'銆俓n"
            f"  Tier-1 涓诲疄楠? wadi, wadi_all15, wadi_variance, batadal, hai\n"
            f"  Tier-2 涓诲疄楠? swat, swat_localfit\n"
            f"  Tier-3 杈呭姪: smd"
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
