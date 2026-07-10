"""
run_baselines.py — Unified Baseline Runner for FedHGF Comparison
================================================================
Produces Prec / Rec / F1 / AUROC for 8 baselines under the
SAME data splits and cal-F1 threshold protocol as FedHGF.

Federated baselines  (same heterogeneous partition):
  fedanomaly   FedAnomaly  (Zhang et al., IPCCC 2021)
  fl_stam      FL-STAM     (Wang et al., ADMA 2025)
  ufedhy       uFedHyAD    (Xie et al.)
  pefad        PeFAD       (Xu et al., KDD 2024)

Centralized baselines (strong non-federated references):
  gdn          GDN         (Deng & Hooi, AAAI 2021)  — per-client local
  mtad_gat     MTAD-GAT    (Zhao et al., KDD 2020)
  tranad       TranAD      (Tuli et al., VLDB 2022)   — centralized mode
  ganf         GANF        (Dai & Chen, ICLR 2022)    — centralized mode

Outputs (per run):
  results_<dataset>_baselines/
    baselines_<dataset>_<ts>.csv          — summary (1 row/method/seed)
    baselines_<dataset>_rounds_<ts>.csv   — per-round (federated only, convergence)

Summary CSV columns:
  dataset, method, type, seed, num_clients, rounds,
  params_million, upload_kb_per_client_round, download_kb_per_client_round,
  total_comm_mb, train_time_min,
  final_precision, final_recall, final_f1, final_auroc, final_auprc,
  macro_auroc, macro_f1

Per-round CSV columns:
  dataset, method, seed, round, precision, recall, f1, auroc, auprc

Usage:
  python baseline/run_baselines.py --dataset batadal_small --label-mode last --device cuda
  python baseline/run_baselines.py --dataset swat --type federated --device cuda
  python baseline/run_baselines.py --dataset wadi --methods fl_stam,gdn --device cuda
  python baseline/run_baselines.py --dataset swat --seeds 42 --quick
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

                                                                                
_this_dir = Path(__file__).resolve().parent
_repo_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))                         
sys.path.insert(0, str(_repo_dir))                                         

from dataset_registry_fair import get_fair_spec               


                                                                                

def _safe_roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _safe_prc_auc(y, s):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def _pick_cal_threshold(s_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Label-free paper protocol: fixed 95th percentile on calibration scores."""
    s = np.asarray(s_cal, dtype=np.float32)
    return float(np.percentile(s, 95))


def _cal_clients(clients: List[dict]) -> List[dict]:
    return [dict(c, X_test=c["X_cal"], y_test=c["y_cal"]) for c in clients]


def _raw_scores(results: List[dict]) -> Dict[str, np.ndarray]:
    """Extract raw_score (or score) dict from predict() output."""
    out = {}
    for r in results:
        name = r.get("client_name", r.get("client_id", "?"))
        s = r.get("raw_score", r.get("score"))
        if s is not None:
            out[str(name)] = np.asarray(s, dtype=np.float64)
    return out


                                                                               

def _count_comm_params(model, method: str) -> int:
    """Count the number of parameters that are communicated each round."""
    try:
        if method == "pefad":
            return sum(p.numel() for p in model.model.parameters()
                       if p.requires_grad)
        if method == "ufedhy":
            return sum(p.numel() for p in model.hypernetwork.parameters())
        return sum(p.numel() for p in model.model.parameters())
    except AttributeError:
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return 0


def _comm_stats(
    model, method: str, n_clients: int, n_rounds: int,
) -> dict:
    """
    Compute communication overhead statistics.

    Upload   = client → server per client per round (float32 bytes → KB)
    Download = server → client per client per round
    For FedAvg variants: upload ≈ download ≈ model_params × 4 bytes.
    Total(MB) = K × R × (upload_KB + download_KB) / 1024
    """
    n_params = _count_comm_params(model, method)
    params_m  = n_params / 1e6
    upload_kb = n_params * 4 / 1024                                         
    download_kb = upload_kb                                        
    total_mb  = n_clients * n_rounds * (upload_kb + download_kb) / 1024
    return dict(
        params_million              = round(params_m, 4),
        upload_kb_per_client_round  = round(upload_kb, 2),
        download_kb_per_client_round= round(download_kb, 2),
        total_comm_mb               = round(total_mb, 3),
    )


def _eval(
    clients: List[dict],
    s_cal: Dict[str, np.ndarray],
    s_te:  Dict[str, np.ndarray],
) -> Tuple[dict, List[dict]]:
    """Cal-F1 threshold → test Prec/Rec/F1/AUROC/AUPRC (pooled + per-client)."""
    from sklearn.metrics import (
        f1_score as _f1, precision_score as _pr, recall_score as _rc)

    all_y, all_s, all_pred = [], [], []
    per_rows = []
    for c in clients:
        name = c["client_name"]
        sc = s_cal.get(name)
        st = s_te.get(name)
        if sc is None or st is None:
            continue
        tau  = _pick_cal_threshold(sc, c["y_cal"])
        y    = np.asarray(c["y_test"], dtype=np.int64)
        n    = min(len(st), len(y))
        st, y = st[:n], y[:n]
        pred = (st > tau).astype(np.int64)
        f1   = float(_f1(y, pred, zero_division=0))
        prec = float(_pr(y, pred, zero_division=0))
        rec  = float(_rc(y, pred, zero_division=0))
        auroc = _safe_roc_auc(y, st)
        auprc = _safe_prc_auc(y, st)
        per_rows.append(dict(
            client=name, auroc=auroc, auprc=auprc,
            f1=f1, precision=prec, recall=rec,
            n_test=n, n_anom=int(y.sum()), tau=tau,
        ))
        all_y.append(y); all_s.append(st); all_pred.append(pred)

    if not all_y:
        empty = dict(auroc=float("nan"), auprc=float("nan"),
                     f1=float("nan"), precision=float("nan"), recall=float("nan"),
                     macro_auroc=float("nan"), macro_f1=float("nan"))
        return empty, []

    y_all    = np.concatenate(all_y)
    s_all    = np.concatenate(all_s)
    pred_all = np.concatenate(all_pred)
    auroc = _safe_roc_auc(y_all, s_all)
    auprc = _safe_prc_auc(y_all, s_all)
    f1   = float(_f1(y_all, pred_all, zero_division=0))
    prec = float(_pr(y_all, pred_all, zero_division=0))
    rec  = float(_rc(y_all, pred_all, zero_division=0))
    macro_auroc = float(mean(r["auroc"] for r in per_rows if not np.isnan(r["auroc"])))
    macro_f1    = float(mean(r["f1"]    for r in per_rows))
    overall = dict(auroc=auroc, auprc=auprc, f1=f1,
                   precision=prec, recall=rec,
                   macro_auroc=macro_auroc, macro_f1=macro_f1)
    return overall, per_rows


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_clients(dataset: str, seed: int, quick: bool = False,
                 window_len_override: int = None,
                 stride_override: int = None,
                 label_mode: str = None):
    import inspect
    spec     = get_fair_spec(dataset)
    loader   = spec["loader"]
    data_dir = spec["default_data_dir"]
    kwargs   = dict(spec["loader_kwargs"])
    kwargs["seed"] = seed
    if window_len_override is not None:
        kwargs["window_len"] = window_len_override
    if stride_override is not None:
        kwargs["stride"] = stride_override
    if label_mode is not None:
        kwargs["label_mode"] = label_mode
    if quick:
        kwargs["max_train_rows"] = kwargs.get("max_train_rows") or 80_000
    sig = inspect.signature(loader).parameters
    has_vkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
    if not has_vkw:
        kwargs = {k: v for k, v in kwargs.items() if k in sig}
    return loader(data_dir=data_dir, **kwargs)


                                                                                
                
                                                                                

_FA_CFG = {
    "swat":    dict(hidden_size=128, kernel_size=3, n_layers=1,
                    n_rounds=10, local_epochs=1, lr=1e-4, batch_size=32,
                    beta_kl=1.0, optimizer="sgd"),
    "wadi":    dict(hidden_size=128, kernel_size=3, n_layers=1,
                    n_rounds=12, local_epochs=1, lr=1e-4, batch_size=32,
                    beta_kl=1.0, optimizer="sgd"),
    "batadal": dict(hidden_size=64,  kernel_size=3, n_layers=1,
                    n_rounds=10, local_epochs=1, lr=1e-4, batch_size=32,
                    beta_kl=1.0, optimizer="sgd"),
    "hai":     dict(hidden_size=128, kernel_size=3, n_layers=1,
                    n_rounds=10, local_epochs=1, lr=1e-4, batch_size=32,
                    beta_kl=1.0, optimizer="sgd"),
}

_FLSTAM_CFG = {
    "swat":    dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, gat_heads=2,
                    dropout=0.1, n_rounds=10, local_epochs=3, batch_size=64,
                    lr=1e-3, lambda_disc=0.1, threshold_mode="ratio"),
    "wadi":    dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, gat_heads=2,
                    dropout=0.1, n_rounds=12, local_epochs=3, batch_size=64,
                    lr=1e-3, lambda_disc=0.1, threshold_mode="ratio"),
    "batadal": dict(d_model=32, n_heads=2, n_layers=2, d_ff=64,  gat_heads=2,
                    dropout=0.1, n_rounds=10, local_epochs=3, batch_size=32,
                    lr=1e-3, lambda_disc=0.1, threshold_mode="ratio"),
    "hai":     dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, gat_heads=2,
                    dropout=0.1, n_rounds=10, local_epochs=3, batch_size=64,
                    lr=1e-3, lambda_disc=0.1, threshold_mode="ratio"),
}

_UFEDHY_CFG = {
    "swat":    dict(seq_len=5, d_model=32, n_heads=4, n_layers=3,
                    comm_rounds=20, local_epochs=1,
                    local_lr=1e-4, hn_lr=1e-3, batch_size=128,
                    emb_dim=100, hidden_dim=100),
    "wadi":    dict(seq_len=5, d_model=32, n_heads=4, n_layers=3,
                    comm_rounds=20, local_epochs=1,
                    local_lr=1e-4, hn_lr=1e-3, batch_size=128,
                    emb_dim=100, hidden_dim=100),
    "batadal": dict(seq_len=5, d_model=32, n_heads=4, n_layers=3,
                    comm_rounds=20, local_epochs=1,
                    local_lr=1e-4, hn_lr=1e-3, batch_size=64,
                    emb_dim=64, hidden_dim=64),
    "hai":     dict(seq_len=5, d_model=32, n_heads=4, n_layers=3,
                    comm_rounds=20, local_epochs=1,
                    local_lr=1e-4, hn_lr=1e-3, batch_size=128,
                    emb_dim=100, hidden_dim=100),
}

_PEFAD_CFG = {
    "swat":    dict(d_model=128, n_layers=6, n_heads=8, d_ff=512, dropout=0.1,
                    n_frozen=4, n_trainable=2, patch_len=4,
                    adms_beta=0.5, mask_ratio=0.25, adms_temperature=0.5,
                    enable_ppds=True, ppds_epochs=15, ppds_synth_size=96,
                    alpha_w=1.0, alpha_mi=0.1, lambda_kd=1.0,
                    n_rounds=10, local_epochs=2, lr=1e-3, batch_size=32),
    "wadi":    dict(d_model=128, n_layers=6, n_heads=8, d_ff=512, dropout=0.1,
                    n_frozen=4, n_trainable=2, patch_len=4,
                    adms_beta=0.5, mask_ratio=0.25, adms_temperature=0.5,
                    enable_ppds=True, ppds_epochs=15, ppds_synth_size=96,
                    alpha_w=1.0, alpha_mi=0.1, lambda_kd=1.0,
                    n_rounds=12, local_epochs=2, lr=1e-3, batch_size=32),
    "batadal": dict(d_model=64,  n_layers=4, n_heads=4, d_ff=256, dropout=0.1,
                    n_frozen=2, n_trainable=2, patch_len=4,
                    adms_beta=0.5, mask_ratio=0.25, adms_temperature=0.5,
                    enable_ppds=True, ppds_epochs=10, ppds_synth_size=64,
                    alpha_w=1.0, alpha_mi=0.1, lambda_kd=1.0,
                    n_rounds=10, local_epochs=2, lr=1e-3, batch_size=32),
    "hai":     dict(d_model=128, n_layers=6, n_heads=8, d_ff=512, dropout=0.1,
                    n_frozen=4, n_trainable=2, patch_len=4,
                    adms_beta=0.5, mask_ratio=0.25, adms_temperature=0.5,
                    enable_ppds=True, ppds_epochs=15, ppds_synth_size=96,
                    alpha_w=1.0, alpha_mi=0.1, lambda_kd=1.0,
                    n_rounds=10, local_epochs=2, lr=1e-3, batch_size=32),
}

_GDN_CFG = {
    "swat":    dict(embed_dim=64,  hidden_dim=64,  topk=15, out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=64,  threshold_mode="ratio"),
    "wadi":    dict(embed_dim=128, hidden_dim=128, topk=30, out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=64,  threshold_mode="ratio"),
    "batadal": dict(embed_dim=32,  hidden_dim=32,  topk=7,  out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=32,  threshold_mode="ratio"),
    "hai":     dict(embed_dim=64,  hidden_dim=64,  topk=15, out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=64,  threshold_mode="ratio"),
}

_MTADGAT_CFG = {
    "swat":    dict(d_gru=128, latent_dim=32, dropout=0.1,
                    n_rounds=10, local_epochs=2, lr=1e-3, batch_size=32,
                    gamma=0.8, kl_weight=0.01),
    "wadi":    dict(d_gru=128, latent_dim=32, dropout=0.1,
                    n_rounds=12, local_epochs=2, lr=1e-3, batch_size=32,
                    gamma=0.8, kl_weight=0.01),
    "batadal": dict(d_gru=64,  latent_dim=16, dropout=0.1,
                    n_rounds=10, local_epochs=2, lr=1e-3, batch_size=32,
                    gamma=0.8, kl_weight=0.01),
    "hai":     dict(d_gru=64,  latent_dim=16, dropout=0.1,
                    n_rounds=10, local_epochs=2, lr=1e-3, batch_size=32,
                    gamma=0.8, kl_weight=0.01),
}

_TRANAD_CFG = {
    "swat":    dict(d_model=64, n_heads=8, d_ff=64, dropout=0.1,
                    n_epochs=15, lr=1e-3, batch_size=128,
                    adv_eps=0.95, use_maml=True, maml_lr=2e-3,
                    federated=False),
    "wadi":    dict(d_model=64, n_heads=8, d_ff=64, dropout=0.1,
                    n_epochs=15, lr=1e-3, batch_size=128,
                    adv_eps=0.95, use_maml=True, maml_lr=2e-3,
                    federated=False),
    "batadal": dict(d_model=64, n_heads=8, d_ff=64, dropout=0.1,
                    n_epochs=15, lr=1e-3, batch_size=128,
                    adv_eps=0.95, use_maml=True, maml_lr=2e-3,
                    federated=False),
    "hai":     dict(d_model=64, n_heads=8, d_ff=64, dropout=0.1,
                    n_epochs=15, lr=1e-3, batch_size=128,
                    adv_eps=0.95, use_maml=True, maml_lr=2e-3,
                    federated=False),
}

_GANF_CFG = {
    "swat":    dict(d_h=64, flow_blocks=6, flow_hidden=32,
                    n_epochs=8, lr=1e-3, lr_A=1e-3, batch_size=128,
                    max_lagrangian_iter=5, federated=False),
    "wadi":    dict(d_h=64, flow_blocks=6, flow_hidden=32,
                    n_epochs=8, lr=1e-3, lr_A=1e-3, batch_size=128,
                    max_lagrangian_iter=5, federated=False),
    "batadal": dict(d_h=32, flow_blocks=4, flow_hidden=32,
                    n_epochs=8, lr=1e-3, lr_A=1e-3, batch_size=128,
                    max_lagrangian_iter=5, federated=False),
    "hai":     dict(d_h=64, flow_blocks=6, flow_hidden=32,
                    n_epochs=8, lr=1e-3, lr_A=1e-3, batch_size=128,
                    max_lagrangian_iter=5, federated=False),
}


                                                                                
                                       
                                                                                

RunResult = Tuple[dict, List[dict], List[dict], dict, dict]
                                                               
                                                                           


def _run_standard(
    tag: str, method_key: str, ModelClass, cfg: dict,
    clients: List[dict],
    n_rounds_key: str = "n_rounds",
    is_federated: bool = True,
) -> RunResult:
    """
    Generic runner with per-round tracking and communication overhead.
      model.fit(clients, round_callback)
      → per-round: score cal+test after each round
      → final: cal-F1 threshold → Prec/Rec/F1/AUROC
      → comm stats: params, upload KB, total MB
    """
    model = ModelClass(**dict(cfg))
    n_rounds = cfg.get(n_rounds_key, cfg.get("comm_rounds", 10))
    n_clients = len(clients)

    per_round_rows: List[dict] = []

    def _round_cb(rnd: int, mdl, clts: List[dict]):
        try:
            cal_r  = mdl.predict(_cal_clients(clts))
            test_r = mdl.predict(clts)
            s_c = _raw_scores(cal_r)
            s_t = _raw_scores(test_r)
            ov, _ = _eval(clts, s_c, s_t)
            per_round_rows.append(dict(round=rnd, **ov))
        except Exception as exc:
            per_round_rows.append(dict(round=rnd,
                                       auroc=float("nan"), auprc=float("nan"),
                                       f1=float("nan"), precision=float("nan"),
                                       recall=float("nan"),
                                       macro_auroc=float("nan"),
                                       macro_f1=float("nan"),
                                       _err=str(exc)))

    t0 = time.time()
    if is_federated:
        model.fit(clients, round_callback=_round_cb)
    else:
        model.fit(clients)
    elapsed_min = (time.time() - t0) / 60.0
    print(f"  [{tag}] fit done ({elapsed_min:.2f} min)")

    cal_res  = model.predict(_cal_clients(clients))
    test_res = model.predict(clients)
    s_cal = _raw_scores(cal_res)
    s_te  = _raw_scores(test_res)
    overall, per_client = _eval(clients, s_cal, s_te)

                                    
    pool_s, pool_y = [], []
    for c in clients:
        name = c["client_name"]
        st = s_te.get(name)
        if st is not None:
            y = np.asarray(c["y_test"], dtype=np.int64)
            n = min(len(st), len(y))
            pool_s.append(st[:n]); pool_y.append(y[:n])
    raw_pooled = {"scores": np.concatenate(pool_s) if pool_s else np.array([]),
                  "y_true": np.concatenate(pool_y) if pool_y else np.array([])}

    if is_federated:
        comm = _comm_stats(model, method_key, n_clients, n_rounds)
    else:
        comm = dict(params_million=float("nan"),
                    upload_kb_per_client_round=0.0,
                    download_kb_per_client_round=0.0,
                    total_comm_mb=0.0)
    comm["train_time_min"] = round(elapsed_min, 3)
    comm["rounds"] = n_rounds
    return overall, per_client, per_round_rows, comm, raw_pooled


def _run_gdn_perclnt(
    tag: str, ModelClass, cfg: dict,
    clients: List[dict],
) -> RunResult:
    """
    GDN runs one model per client (local, no federation).
    Train on client's X_train; score X_cal and X_test separately.
    """
    t0 = time.time()
    s_cal: Dict[str, np.ndarray] = {}
    s_te:  Dict[str, np.ndarray] = {}
    last_model = None
    for c in clients:
        m = ModelClass(**dict(cfg))
        c_cal = dict(c, X_test=c["X_cal"], y_test=c["y_cal"])
        if hasattr(m, "fit_single"):
            m.fit_single(c)
            s_cal[c["client_name"]] = np.asarray(
                m.predict_single(c_cal), dtype=np.float64)
            s_te[c["client_name"]]  = np.asarray(
                m.predict_single(c),    dtype=np.float64)
        else:
            m.fit([c])
            cal_r = m.predict([c_cal])
            te_r  = m.predict([c])
            s_cal[c["client_name"]] = _raw_scores(cal_r).get(
                c["client_name"],
                np.asarray(cal_r[0].get("raw_score", cal_r[0].get("score")),
                           dtype=np.float64))
            s_te[c["client_name"]] = _raw_scores(te_r).get(
                c["client_name"],
                np.asarray(te_r[0].get("raw_score", te_r[0].get("score")),
                           dtype=np.float64))
        last_model = m
        print(f"    [{tag}] {c['client_name']} scored")
    elapsed_min = (time.time() - t0) / 60.0
    overall, per_client = _eval(clients, s_cal, s_te)

                                    
    pool_s, pool_y = [], []
    for c in clients:
        name = c["client_name"]
        st = s_te.get(name)
        if st is not None:
            y = np.asarray(c["y_test"], dtype=np.int64)
            n = min(len(st), len(y))
            pool_s.append(st[:n]); pool_y.append(y[:n])
    raw_pooled = {"scores": np.concatenate(pool_s) if pool_s else np.array([]),
                  "y_true": np.concatenate(pool_y) if pool_y else np.array([])}

    comm = dict(params_million=float("nan"),
                upload_kb_per_client_round=0.0,
                download_kb_per_client_round=0.0,
                total_comm_mb=0.0,
                train_time_min=round(elapsed_min, 3),
                rounds=0)
    return overall, per_client, [], comm, raw_pooled


                                                                                
                         
                                                                                

FEDERATED_METHODS   = ["fedanomaly", "fl_stam", "ufedhy", "pefad"]
CENTRALIZED_METHODS = ["gdn", "mtad_gat", "tranad", "ganf"]
ALL_METHODS = FEDERATED_METHODS + CENTRALIZED_METHODS


def run_one(
    method: str,
    clients: List[dict],
    ds_key:  str,
    device:  str,
    seed:    int,
) -> RunResult:
    """Dispatch to the right model runner. Returns (overall, per_client, per_round, comm)."""

    def _cfg(table):
        c = dict(table.get(ds_key, table.get("swat", {})))
        c["device"] = device
        c["seed"]   = seed
        return c

    if method == "fedanomaly":
        from fedanomaly_model import FedAnomaly
        return _run_standard("FedAnomaly", "fedanomaly", FedAnomaly,
                             _cfg(_FA_CFG), clients, is_federated=True)

    if method == "fl_stam":
        from fl_stam_model import FLSTAM
        return _run_standard("FL-STAM", "fl_stam", FLSTAM,
                             _cfg(_FLSTAM_CFG), clients, is_federated=True)

    if method == "ufedhy":
        from ufedhy_model import uFedHyDisMTSADD
        cfg = _cfg(_UFEDHY_CFG)
        cfg["threshold_mode"] = "ratio"
        cfg["client_sample_rate"] = 1.0
        return _run_standard("uFedHyAD", "ufedhy", uFedHyDisMTSADD,
                             cfg, clients, n_rounds_key="comm_rounds",
                             is_federated=True)

    if method == "pefad":
        from pefad_model import PeFAD
        cfg = _cfg(_PEFAD_CFG)
        cfg["threshold_mode"] = "ratio"
        return _run_standard("PeFAD", "pefad", PeFAD,
                             cfg, clients, is_federated=True)

    if method == "gdn":
        from gdn_model import GDN
        return _run_gdn_perclnt("GDN", GDN, _cfg(_GDN_CFG), clients)

    if method == "mtad_gat":
        from mtad_gat_model import MtadGAT
        cfg = _cfg(_MTADGAT_CFG)
        cfg["threshold_mode"] = "ratio"
        return _run_standard("MTAD-GAT", "mtad_gat", MtadGAT,
                             cfg, clients, is_federated=False)

    if method == "tranad":
        from tranad_model import TranAD
        cfg = _cfg(_TRANAD_CFG)
        cfg["threshold_mode"] = "ratio"
        return _run_standard("TranAD", "tranad", TranAD,
                             cfg, clients, is_federated=False)

    if method == "ganf":
        from ganf_model import GANF
        cfg = _cfg(_GANF_CFG)
        cfg["threshold_mode"] = "ratio"
        return _run_standard("GANF", "ganf", GANF,
                             cfg, clients, is_federated=False)

    raise ValueError(f"Unknown method: {method}")


                                                                                
                
                                                                                

_METHOD_LABEL = {
    "fedanomaly": "FedAnomaly",
    "fl_stam":    "FL-STAM",
    "ufedhy":     "uFedHyAD",
    "pefad":      "PeFAD",
    "gdn":        "GDN",
    "mtad_gat":   "MTAD-GAT",
    "tranad":     "TranAD",
    "ganf":       "GANF",
}
_METHOD_TYPE = {m: "Federated"   for m in FEDERATED_METHODS}
_METHOD_TYPE.update({m: "Centralized" for m in CENTRALIZED_METHODS})


def _print_summary(dataset: str, rows: List[dict]):
    print()
    print("=" * 78)
    print(f"  Baseline Summary — {dataset.upper()}")
    print("=" * 78)
    hdr = f"  {'Type':<13} {'Method':<14}  {'AUROC':>7}  {'AUPRC':>7}  "
    hdr += f"{'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'MacroF1':>8}"
    print(hdr)
    print("  " + "-" * 74)
    def _g(r, *keys):
        for k in keys:
            v = r.get(k)
            if v is not None:
                return float(v)
        return float("nan")

    for r in rows:
        line = (f"  {r.get('type','?'):<13} {r.get('method','?'):<14}  "
                f"{_g(r,'final_auroc','auroc'):>7.4f}  "
                f"{_g(r,'final_auprc','auprc'):>7.4f}  "
                f"{_g(r,'final_precision','precision'):>7.4f}  "
                f"{_g(r,'final_recall','recall'):>7.4f}  "
                f"{_g(r,'final_f1','f1'):>7.4f}  "
                f"{_g(r,'macro_f1'):>8.4f}")
        print(line)
    print("=" * 78)


def _save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    all_keys: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


                                                                                
      
                                                                                

def main():
    ap = argparse.ArgumentParser(
        description="Unified baseline runner for FedHGF comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset",  required=True,
                    choices=["swat", "wadi", "batadal", "batadal_small", "hai"],
                    help="Dataset to benchmark")
    ap.add_argument("--methods",  default=None,
                    help="Comma-separated method names, e.g. fedanomaly,gdn. "
                         "Overrides --type.")
    ap.add_argument("--type",     default="all",
                    choices=["all", "federated", "centralized"],
                    help="Method group to run")
    ap.add_argument("--device",   default="cuda")
    ap.add_argument("--seeds",    default="42",
                    help="Comma-separated seeds, e.g. 42 or 42,123")
    ap.add_argument("--quick",    action="store_true",
                    help="Use smaller training subset for quick validation")
    ap.add_argument("--window-len", type=int, default=None,
                    help="Override window_len in data loader (e.g. 32 for BATADAL B33.3)")
    ap.add_argument("--stride",     type=int, default=None,
                    help="Override stride in data loader")
    ap.add_argument("--label-mode", default=None,
                    choices=["any", "last", "center", "majority"],
                    help="Override label_mode in data loader (e.g. last for BATADAL)")
    ap.add_argument("--out-dir",  default=None,
                    help="Directory to save CSV results")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    elif args.type == "federated":
        methods = FEDERATED_METHODS
    elif args.type == "centralized":
        methods = CENTRALIZED_METHODS
    else:
        methods = ALL_METHODS

    invalid = [m for m in methods if m not in ALL_METHODS]
    if invalid:
        ap.error(f"Unknown methods: {invalid}. Choose from {ALL_METHODS}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _repo_dir / f"results_{args.dataset}_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print(f"  Baseline Runner | dataset={args.dataset} | methods={methods}")
    print(f"  seeds={seeds} | device={args.device} | quick={args.quick}")
    if args.window_len or args.label_mode or args.stride:
        print(f"  [Protocol Override] window_len={args.window_len}  "
              f"stride={args.stride}  label_mode={args.label_mode}")
    print("=" * 78)

    ds_key = args.dataset.split("_")[0]
    summary_rows:   List[dict] = []
    per_round_rows: List[dict] = []
    raw_scores_all: dict = {}                                                  

    for method in methods:
        seed_results:    List[dict] = []
        seed_comm:       List[dict] = []

        for seed in seeds:
            print(f"\n{'─'*60}")
            print(f"  [{_METHOD_LABEL[method]}] seed={seed}")
            print(f"{'─'*60}")
            set_global_seed(seed)
            clients, n_anchor, _ = load_clients(
                args.dataset, seed, quick=args.quick,
                window_len_override=args.window_len,
                stride_override=args.stride,
                label_mode=args.label_mode)
            ds_key = args.dataset.split("_")[0]
            print(f"  data loaded: {len(clients)} clients, n_anchor={n_anchor}")

            try:
                overall, _per, rounds, comm, raw_pooled = run_one(
                    method, clients, ds_key, args.device, seed)

                print(f"  [{_METHOD_LABEL[method]}] "
                      f"AUROC={overall['auroc']:.4f}  "
                      f"AUPRC={overall['auprc']:.4f}  "
                      f"Prec={overall['precision']:.4f}  "
                      f"Rec={overall['recall']:.4f}  "
                      f"F1={overall['f1']:.4f}  "
                      f"MacroF1={overall['macro_f1']:.4f}  "
                      f"| CommMB={comm.get('total_comm_mb', 0):.2f}  "
                      f"time={comm.get('train_time_min', 0):.2f}min")

                seed_results.append(overall)
                seed_comm.append(comm)
                                                                 
                if seed == seeds[0] and raw_pooled["scores"].size > 0:
                    raw_scores_all[_METHOD_LABEL[method]] = raw_pooled

                for rr in rounds:
                    per_round_rows.append(dict(
                        dataset=args.dataset, method=_METHOD_LABEL[method],
                        seed=seed, **rr))

            except Exception as exc:
                import traceback
                print(f"  [ERROR] {_METHOD_LABEL[method]} seed={seed}: {exc}")
                traceback.print_exc()
                continue

        if not seed_results:
            continue

        def _avg_ov(key):
            vals = [r[key] for r in seed_results
                    if not (isinstance(r[key], float) and np.isnan(r[key]))]
            return round(float(mean(vals)), 6) if vals else float("nan")

        def _avg_cm(key):
            vals = [c[key] for c in seed_comm
                    if not (isinstance(c[key], float) and np.isnan(c[key]))]
            return round(float(mean(vals)), 4) if vals else float("nan")

        comm0 = seed_comm[0] if seed_comm else {}
        summary_rows.append(dict(
            dataset                      = args.dataset,
            method                       = _METHOD_LABEL[method],
            type                         = _METHOD_TYPE[method],
            seed                         = args.seeds,
            num_clients                  = len(clients),
            rounds                       = comm0.get("rounds", 0),
            params_million               = _avg_cm("params_million"),
            upload_kb_per_client_round   = _avg_cm("upload_kb_per_client_round"),
            download_kb_per_client_round = _avg_cm("download_kb_per_client_round"),
            total_comm_mb                = _avg_cm("total_comm_mb"),
            train_time_min               = _avg_cm("train_time_min"),
            final_precision              = _avg_ov("precision"),
            final_recall                 = _avg_ov("recall"),
            final_f1                     = _avg_ov("f1"),
            final_auroc                  = _avg_ov("auroc"),
            final_auprc                  = _avg_ov("auprc"),
            macro_auroc                  = _avg_ov("macro_auroc"),
            macro_f1                     = _avg_ov("macro_f1"),
        ))

    summary_path    = out_dir / f"baselines_{args.dataset}_{ts}.csv"
    per_round_path  = out_dir / f"baselines_{args.dataset}_rounds_{ts}.csv"
    _save_csv(summary_rows,   summary_path)
    _save_csv(per_round_rows, per_round_path)
    print(f"\n  Summary  → {summary_path}")
    print(f"  Per-round → {per_round_path}  ({len(per_round_rows)} rows)")

                                            
    if raw_scores_all:
        npz_data = {}
        for label, rp in raw_scores_all.items():
            safe_key = label.lower().replace("-", "_").replace(" ", "_")
            npz_data[f"s_{safe_key}"] = rp["scores"]
            npz_data[f"y_{safe_key}"] = rp["y_true"]
        scores_path = out_dir / f"raw_scores_{ds_key}.npz"
        np.savez(scores_path, **npz_data)
        print(f"  Raw scores → {scores_path}  ({list(npz_data.keys())})")
    _print_summary(args.dataset, summary_rows)


if __name__ == "__main__":
    main()
