"""
FedHGF fair benchmark runner.

Examples:
  python run_fair_comparison.py --dataset wadi --method fedhgf --seeds 42 --device cuda --w-fusion 0.20,0.55,0.25
  python run_fair_comparison.py --dataset hai --method fedhgf --seeds 42,123,2024 --device cuda --w-fusion 0.40,0.30,0.30 --threshold-mode f1_rate_guard --lambda-c 0.005 --lambda-v 0.4
  python run_fair_comparison.py --list-datasets
  python run_fair_comparison.py --dataset batadal_small --info-only

Outputs:
  - results_<dataset>_fair/fair_<dataset>_<timestamp>.csv
  - results_<dataset>_fair/per_client_<dataset>_<timestamp>.csv
  - results_<dataset>_fair/input_table_<dataset>_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np

            
_this = Path(__file__).resolve().parent
_root = _this.parent
_new  = _root / "New"
sys.path.insert(0, str(_this))
sys.path.insert(1, str(_root))
if str(_new) not in sys.path:
    sys.path.append(str(_new))

from dataset_registry_fair import (
    get_fair_spec, BASE_CFG_FAIR, DATASET_REGISTRY_FAIR,
    list_datasets_by_tier, BASELINE_CATEGORIES,
)

                                                                        

def _safe_roc_auc(y_true, scores):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_prc_auc(y_true, scores):
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _oracle_threshold_f1(y_true, scores):
    """Oracle upper-bound F1: sweeps test-set threshold (uses test labels — diagnostics only)."""
    best_f1, best_thr = 0.0, 0.5
    for pct in np.linspace(50, 99, 50):
        thr = float(np.percentile(scores, pct))
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    pred = (scores >= best_thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    return best_f1, prec, rec


def pick_cal_f1_threshold(scores_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Label-free paper protocol: fixed 95th percentile on calibration scores."""
    s = np.asarray(scores_cal, dtype=np.float32)
    return float(np.percentile(s, 95))


def _make_cal_clients(clients: List[dict]) -> List[dict]:
    """Return a shallow copy with X_test/y_test replaced by X_cal/y_cal."""
    return [dict(c, X_test=c["X_cal"], y_test=c["y_cal"]) for c in clients]


def _calibrated_eval(
    clients: List[dict],
    score_cal_dict: Dict[str, np.ndarray],
    score_test_dict: Dict[str, np.ndarray],
) -> Tuple[float, float, float, float, float, Dict[str, np.ndarray]]:
    """Per-client cal-F1 threshold → test predictions → pooled metrics."""
    from sklearn.metrics import (
        f1_score as _f1s, precision_score as _prs, recall_score as _rcs)
    all_y, all_s, all_pred = [], [], []
    pred_dict: Dict[str, np.ndarray] = {}
    for c in clients:
        name  = c["client_name"]
        s_cal = score_cal_dict.get(name)
        s_te  = score_test_dict.get(name)
        y_te  = c["y_test"]
        if s_cal is None or s_te is None:
            continue
        tau  = pick_cal_f1_threshold(s_cal, c["y_cal"])
        n    = min(len(s_te), len(y_te))
        pred = (s_te[:n] > tau).astype(np.int64)
        pred_dict[name] = pred
        all_y.append(y_te[:n])
        all_s.append(s_te[:n])
        all_pred.append(pred)
    y_all    = np.concatenate(all_y)
    s_all    = np.concatenate(all_s)
    pred_all = np.concatenate(all_pred)
    auroc = _safe_roc_auc(y_all, s_all)
    auprc = _safe_prc_auc(y_all, s_all)
    f1    = float(_f1s(y_all, pred_all, zero_division=0))
    prec  = float(_prs(y_all, pred_all, zero_division=0))
    rec   = float(_rcs(y_all, pred_all, zero_division=0))
    return auroc, auprc, f1, prec, rec, pred_dict


def compute_per_client_metrics(
    clients: List[dict],
    score_dict: Dict[str, np.ndarray],
    pred_dict:    Dict[str, np.ndarray]       = None,
    tau_dict:     Dict[str, float]            = None,
    sel_w_dict:   Dict[str, tuple]            = None,
    cal_f1_dict:  Dict[str, float]            = None,
) -> List[dict]:
    """
    score_dict : {client_name -> anomaly scores (1-D, len = n_test_windows)}
    pred_dict  : {client_name -> binary predictions} from calibrated threshold.
                 If None, falls back to oracle threshold sweep (for diagnostics).
    """
    from sklearn.metrics import (
        f1_score as _f1s, precision_score as _prs, recall_score as _rcs)
    rows = []
    for c in clients:
        name   = c["client_name"]
        scores = score_dict.get(name)
        if scores is None:
            continue
        y = c["y_test"]
        n = min(len(scores), len(y))
        scores, y = scores[:n], y[:n]
        auroc = _safe_roc_auc(y, scores)
        auprc = _safe_prc_auc(y, scores)
        if pred_dict is not None and name in pred_dict:
            pred = pred_dict[name][:n]
            f1   = float(_f1s(y, pred, zero_division=0))
            prec = float(_prs(y, pred, zero_division=0))
            rec  = float(_rcs(y, pred, zero_division=0))
        else:
            f1, prec, rec = _oracle_threshold_f1(y, scores)                     
        pred_rate = float(pred[:n].mean()) if pred_dict is not None and name in pred_dict else float("nan")
        true_rate = float(y.mean())
        cal_anom  = int(c.get("y_cal", np.zeros(1)).sum())
        cal_tot   = len(c.get("y_cal", np.zeros(1)))
        cal_rate  = cal_anom / max(1, cal_tot)
        tau      = tau_dict.get(name, float("nan")) if tau_dict else float("nan")
        sel_w    = sel_w_dict.get(name)  if sel_w_dict  else None
        cal_f1_v = cal_f1_dict.get(name, float("nan")) if cal_f1_dict else float("nan")
        rows.append({
            "client":      name,
            "n_k":         c.get("n_k", c["X_test"].shape[2]),
            "n_test":      len(y),
            "n_anom":      int(y.sum()),
            "auroc":       auroc,
            "auprc":       auprc,
            "f1":          f1,
            "precision":   prec,
            "recall":      rec,
            "pred_rate":   pred_rate,
            "true_rate":   true_rate,
            "cal_rate":    cal_rate,
            "tau":         tau,
            "w1":          float(sel_w[0]) if sel_w else float("nan"),
            "w2":          float(sel_w[1]) if sel_w else float("nan"),
            "w3":          float(sel_w[2]) if sel_w and len(sel_w) > 2 else float("nan"),
            "cal_f1_sel":  cal_f1_v,
        })
    return rows


def macro_from_per_client(rows: List[dict]) -> dict:
    """Macro-average AUROC, AUPRC, F1 across clients (each client weighted equally)."""
    def _m(key):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(mean(vals)) if vals else float("nan")
    return {
        "macro_auroc": _m("auroc"),
        "macro_auprc": _m("auprc"),
        "macro_f1":    _m("f1"),
        "macro_prec":  _m("precision"),
        "macro_rec":   _m("recall"),
        "n_clients":   len(rows),
    }


                                                                        

def set_seed(seed: int):
    import random
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    random.seed(seed)
    np.random.seed(seed)


def load_clients(dataset: str, seed: int, quick: bool = False,
                 label_mode: Optional[str] = None):
    spec     = get_fair_spec(dataset)
    loader   = spec["loader"]
    data_dir = spec["default_data_dir"]
    kwargs   = dict(spec["loader_kwargs"])
    kwargs["seed"] = seed
    if label_mode is not None:
        kwargs["label_mode"] = label_mode
    if quick:
        kwargs["max_train_rows"] = kwargs.get("max_train_rows") or 80_000
    sig = inspect.signature(loader).parameters
    has_vkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
    if not has_vkw:
        kwargs = {k: v for k, v in kwargs.items() if k in sig}
    return loader(data_dir=data_dir, **kwargs)


def _collect_scores(
    model, clients: List[dict],
) -> Tuple[Dict[str, np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Unified score collector that handles two model interfaces:
      (a) model.predict(clients) -> List[dict] with 'client_name','score','y_true'
      (b) model.predict_client(c) -> np.ndarray of anomaly scores
    Returns (score_dict, all_y, all_s).
    """
                             
    if hasattr(model, "predict") and not hasattr(model, "predict_client"):
        raw = model.predict(clients)
        score_dict = {r["client_name"]: r["score"] for r in raw}
        all_y = [r["y_true"][:len(r["score"])] for r in raw]
        all_s = [r["score"][:len(r["y_true"])] for r in raw]
        return score_dict, all_y, all_s
                   
    score_dict: Dict[str, np.ndarray] = {}
    all_y, all_s = [], []
    for c in clients:
        s = model.predict_client(c)
        score_dict[c["client_name"]] = s
        y = c["y_test"]
        n = min(len(s), len(y))
        all_y.append(y[:n])
        all_s.append(s[:n])
    return score_dict, all_y, all_s


def print_feature_report(clients: List[dict], method_name: str):
    print(f"\n  [特征维度报告 — {method_name}]")
    total = 0
    for c in clients:
        n_a = len(c.get("anchor_names", []))
        n_x = len(c.get("aux_names", []))
        n_k = c.get("n_k", c["X_train"].shape[2])
        total += n_k
        print(f"    {c['client_name']}: n_k={n_k} (anchor={n_a}, aux={n_x})")
    print(f"    总维度（所有客户端之和）: {total}")
    print()


def input_transparency_row(
    dataset: str, method: str, clients: List[dict],
    protocol: str, aggregated_params: str,
) -> dict:
    """Build one row for the input transparency table (paper appendix)."""
    dims = [c.get("n_k", c["X_train"].shape[2]) for c in clients]
    return {
        "dataset": dataset,
        "method":  method,
        "n_clients": len(clients),
        "client_dims": "/".join(map(str, dims)),
        "input_protocol": protocol,
        "aggregated_params": aggregated_params,
    }


def save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  保存: {path}")


def save_results(results_dict: dict, out_dir: str, stem: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_dir) / f"{stem}.csv"
    if not results_dict:
        return
    keys = list(next(iter(results_dict.values())).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + keys)
        w.writeheader()
        for method, row in results_dict.items():
            w.writerow({"method": method, **row})
    print(f"  结果已保存: {csv_path}")
    return csv_path


                                                                            

_FEDHGF_CFG_BASE = {
    "swat":    dict(d_h=64, flow_hidden=128),
    "wadi":    dict(d_h=64, flow_hidden=128, collapse_std_thr=0.003),
    "batadal": dict(d_h=64, collapse_std_thr=0.003),
    "hai":     dict(d_h=64, flow_hidden=128),
    "smd":     dict(d_h=64),
}


def run_fedhgf(
    clients, n_anchor: int, dataset: str,
    seed: int, fedhgf_full: bool, device: str,
    w_fusion: tuple = None,
    threshold_mode: str = None,
    target_anom_rate: float = None,
    fusion_mode: str = None,
    cfg_extra: dict = None,
    stage1_round_cb=None,
    return_round_history: bool = False,
    fusion_candidates_extra: list = None,
) -> Tuple[dict, dict]:
    """Returns (overall_metrics, per_client_scores)."""
    from fedgad_full import FedGAD

    ds_key = dataset.split("_")[0]                      
    spec = get_fair_spec(dataset)
    cfg  = dict(BASE_CFG_FAIR)
    cfg.update(_FEDHGF_CFG_BASE.get(ds_key, {}))
    cfg.update(spec.get("cfg_overrides", {}))
    if fedhgf_full:
        cfg.update(spec.get("cfg_overrides_full", {}))
        cfg["use_label_assisted_fusion"] = True

    if w_fusion is not None:
        cfg["w_fusion"] = tuple(w_fusion)
    if threshold_mode is not None:
        cfg["adaptive_threshold_mode"] = threshold_mode
    if target_anom_rate is not None:
        cfg["target_anom_rate"] = target_anom_rate
    if fusion_mode is not None:
        cfg["fusion_mode"] = fusion_mode
    if cfg_extra:
        cfg.update(cfg_extra)
    cfg["device"] = device
    cfg["seed"]   = seed

    label = "FedHGF-Full" if fedhgf_full else "FedHGF"
    print(f"\n  [{label}] n_anchor={n_anchor}  "
          f"use_label_fusion={cfg['use_label_assisted_fusion']}  "
          f"w_fusion={cfg['w_fusion']}")
    print(f"  [{label}] use_prediction_loss={cfg.get('use_prediction_loss', False)}  "
          f"lambda_pred={cfg.get('lambda_pred', 1.0)}  "
          f"lambda_c={cfg.get('lambda_c', 0.02)}  "
          f"center_score_mode={cfg.get('center_score_mode', 'global')}  "
          f"fusion_mode={cfg.get('fusion_mode', 'fixed')}")

    model = FedGAD(n_anchor=n_anchor, **cfg)
    if fusion_candidates_extra:
        model.cfg["fusion_small_candidates_extra"] = fusion_candidates_extra
    model.fit(clients, stage1_round_cb=stage1_round_cb)

                                                                             
    raw_results = model.predict(clients)
    score_dict: Dict[str, np.ndarray] = {
        r["client_name"]: r["score"] for r in raw_results
    }
    all_y, all_s = [], []
    for r in raw_results:
        s = r["score"]
        y = r["y_true"]
        n = min(len(s), len(y))
        all_y.append(y[:n])
        all_s.append(s[:n])

                           
    y_all  = np.concatenate(all_y)
    s_all  = np.concatenate(all_s)
    auroc  = _safe_roc_auc(y_all, s_all)
    auprc  = _safe_prc_auc(y_all, s_all)

                                                                                   
    all_pred = []
    for r in raw_results:
        yp = r["y_pred"]
        yt = r["y_true"]
        n  = min(len(yp), len(yt))
        all_pred.append(yp[:n])
    pred_all = np.concatenate(all_pred)
    from sklearn.metrics import (
        f1_score as _f1s, precision_score as _prs, recall_score as _rcs)
    f1_cal   = float(_f1s(y_all, pred_all, zero_division=0))
    prec_cal = float(_prs(y_all, pred_all, zero_division=0))
    rec_cal  = float(_rcs(y_all, pred_all, zero_division=0))

                                                                             
    branch_info: Dict[str, float] = {}
    try:
        s1_list, s2_list, s3_list = [], [], []
        for r in raw_results:
            yt = r["y_true"]
            n  = len(yt)
            if r.get("s1_raw") is not None:
                s1_list.append(r["s1_raw"][:n])
            if r.get("s2_raw") is not None:
                s2_list.append(r["s2_raw"][:n])
            if r.get("s3_raw") is not None:
                s3_list.append(r["s3_raw"][:n])
        if s1_list:
            branch_info["auroc_s1"] = _safe_roc_auc(y_all, np.concatenate(s1_list))
        if s2_list:
            branch_info["auroc_s2"] = _safe_roc_auc(y_all, np.concatenate(s2_list))
        if s3_list:
            branch_info["auroc_s3"] = _safe_roc_auc(y_all, np.concatenate(s3_list))
    except Exception:
        pass

    pred_dict    = {r["client_name"]: r["y_pred"]                   for r in raw_results}
    tau_dict     = {r["client_name"]: r.get("tau",        float("nan")) for r in raw_results}
    sel_w_dict   = {r["client_name"]: r.get("selected_w")             for r in raw_results}
    cal_f1_dict  = {r["client_name"]: r.get("cal_f1",    float("nan")) for r in raw_results}
    per   = compute_per_client_metrics(
        clients, score_dict, pred_dict, tau_dict,
        sel_w_dict=sel_w_dict, cal_f1_dict=cal_f1_dict)
    macro = macro_from_per_client(per)

    pred_rate_all = float(pred_all.mean())
    true_rate_all = float(y_all.mean())
    print(f"  [{label}] pooled AUROC={auroc:.4f} AUPRC={auprc:.4f} "
          f"Prec={prec_cal:.4f} Rec={rec_cal:.4f} F1_cal={f1_cal:.4f} "
          f"pred_rate={pred_rate_all:.4f} true_rate={true_rate_all:.4f}")
    print(f"           macro-AUROC={macro['macro_auroc']:.4f} macro-F1={macro['macro_f1']:.4f}")
    if branch_info:
        bstr = "  ".join(f"{k}={v:.4f}" for k, v in branch_info.items())
        print(f"           branches: {bstr}")

                                                                   
    try:
        from sklearn.metrics import (
            f1_score as _f1b, precision_score as _prb, recall_score as _rcb)
        for bname in ("s1", "s2", "s3"):
            b_cal_all, yc_all, b_te_all, yt_all = [], [], [], []
            for k, r in enumerate(raw_results):
                b_cal = model.client_cal.get(k, {}).get(bname)
                b_te  = r.get(f"{bname}_raw")
                yc    = clients[k].get("y_cal")
                yt    = r["y_true"]
                if b_cal is None or b_te is None or yc is None:
                    continue
                nc = min(len(b_cal), len(yc))
                nt = min(len(b_te),  len(yt))
                b_cal_all.append(b_cal[:nc]); yc_all.append(yc[:nc])
                b_te_all.append(b_te[:nt]);   yt_all.append(yt[:nt])
            if not b_cal_all:
                continue
            E_cal_b = np.concatenate(b_cal_all)
            y_cal_b = np.concatenate(yc_all)
            E_te_b  = np.concatenate(b_te_all)
            y_te_b  = np.concatenate(yt_all)
            btau, bf1 = float(np.quantile(E_cal_b, 0.95)), -1.0
            if len(np.unique(y_cal_b)) >= 2:
                for q in np.linspace(0.01, 0.99, 99):
                    tau = float(np.quantile(E_cal_b, q))
                    yp  = (E_cal_b > tau).astype(np.int64)
                    f1v = float(_f1b(y_cal_b, yp, zero_division=0))
                    if f1v > bf1:
                        bf1, btau = f1v, tau
            yp_b = (E_te_b > btau).astype(np.int64)
            f1_b  = float(_f1b(y_te_b,  yp_b, zero_division=0))
            pr_b  = float(_prb(y_te_b,  yp_b, zero_division=0))
            rc_b  = float(_rcb(y_te_b,  yp_b, zero_division=0))
            pr_rate = float(yp_b.mean())
            print(f"           [{bname}-only] Prec={pr_b:.4f} Rec={rc_b:.4f} "
                  f"F1={f1_b:.4f} pred_rate={pr_rate:.4f}")
    except Exception:
        pass
    print(f"  {'Client':<14} {'w1':>5} {'w2':>5} {'w3':>5} "
          f"{'calF1':>6} {'tau':>8} {'cal%':>6} {'test%':>6} "
          f"{'pred%':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    for row in per:
        tau_s = f"{row['tau']:.4f}" if not np.isnan(row['tau']) else '  N/A '
        w1_s  = f"{row['w1']:.2f}"  if not np.isnan(row['w1'])  else '  -- '
        w2_s  = f"{row['w2']:.2f}"  if not np.isnan(row['w2'])  else '  -- '
        w3_s  = f"{row['w3']:.2f}"  if not np.isnan(row['w3'])  else '  -- '
        cf_s  = f"{row['cal_f1_sel']:.3f}" if not np.isnan(row['cal_f1_sel']) else ' --- '
        print(f"  {row['client']:<14} {w1_s:>5} {w2_s:>5} {w3_s:>5} "
              f"{cf_s:>6} {tau_s:>8} {row['cal_rate']*100:>5.1f}% "
              f"{row['true_rate']*100:>5.1f}% {row['pred_rate']*100:>5.1f}% "
              f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f}")

    overall = {
        "auroc": auroc, "auprc": auprc,
        "f1": f1_cal, "precision": prec_cal, "recall": rec_cal,
        **macro, **branch_info,
        "label_fusion": cfg["use_label_assisted_fusion"],
        "category": "FedHGF",
    }
    if return_round_history:
        return overall, per, list(model.round_history)
    return overall, per


                                                                           

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
    "smd":     dict(d_model=32, n_heads=2, n_layers=2, d_ff=64,  gat_heads=2,
                    dropout=0.1, n_rounds=10, local_epochs=3, batch_size=64,
                    lr=1e-3, lambda_disc=0.1, threshold_mode="ratio"),
}


def run_fl_stam(
    clients, dataset: str, seed: int, device: str,
) -> Tuple[dict, dict]:
    sys.path.insert(0, str(_root / "FL-STAM"))
    from fl_stam_model import FLSTAM

    ds_key = dataset.split("_")[0]
    cfg = dict(_FLSTAM_CFG.get(ds_key, _FLSTAM_CFG["swat"]))
    cfg["device"] = device
    cfg["seed"]   = seed

    dims = [c.get("n_k", c["X_train"].shape[2]) for c in clients]
    print(f"\n  [FL-STAM] 客户端维度: {dims}")
    print(f"  [FL-STAM] 输入协议: anchor+aux padded to max={max(dims)}")
    print(f"  [FL-STAM] 联邦参数: full backbone aggregated via FedAvg")

    model = FLSTAM(**cfg)
    model.fit(clients)
                                                                    
    cal_clients  = _make_cal_clients(clients)
    s_cal_dict, _, _ = _collect_scores(model, cal_clients)
    s_te_dict, _, _  = _collect_scores(model, clients)
    auroc, auprc, f1, prec, rec, pred_dict = _calibrated_eval(
        clients, s_cal_dict, s_te_dict)
    per   = compute_per_client_metrics(clients, s_te_dict, pred_dict)
    macro = macro_from_per_client(per)

    print(f"  [FL-STAM] AUROC={auroc:.4f} AUPRC={auprc:.4f} F1_cal={f1:.4f} "
          f"| macro-AUROC={macro['macro_auroc']:.4f}")

    overall = {
        "auroc": auroc, "auprc": auprc, "f1": f1,
        "precision": prec, "recall": rec,
        **macro,
        "label_fusion": False,
        "category": "Federated",
    }
    return overall, per


                                                                           

_GDN_CFG = {
    "swat":    dict(embed_dim=64, hidden_dim=64, topk=15, out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=64, threshold_mode="ratio"),
    "wadi":    dict(embed_dim=128, hidden_dim=128, topk=30, out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=64, threshold_mode="ratio"),
    "batadal": dict(embed_dim=32, hidden_dim=32, topk=7,  out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=32, threshold_mode="ratio"),
    "hai":     dict(embed_dim=64, hidden_dim=64, topk=15, out_layers=2,
                    n_epochs=30, lr=1e-3, batch_size=64, threshold_mode="ratio"),
    "smd":     dict(embed_dim=32, hidden_dim=32, topk=8,  out_layers=2,
                    n_epochs=25, lr=1e-3, batch_size=64, threshold_mode="ratio"),
}


def run_gdn(
    clients, dataset: str, seed: int, device: str,
) -> Tuple[dict, dict]:
    sys.path.insert(0, str(_root / "GDN"))
    from gdn_model import GDN

    ds_key = dataset.split("_")[0]
    cfg = dict(_GDN_CFG.get(ds_key, _GDN_CFG["swat"]))
    cfg["device"] = device
    cfg["seed"]   = seed

    dims = [c.get("n_k", c["X_train"].shape[2]) for c in clients]
    print(f"\n  [GDN] 客户端维度: {dims}")
    print(f"  [GDN] 输入协议: per-client anchor+aux (no federation, local training)")

    def _gdn_scores(m, train_c, eval_c):
        """Train on train_c, score eval_c (handles both fit_single and fit APIs)."""
        if hasattr(m, "fit_single"):
            m.fit_single(train_c)
            return m.predict_single(eval_c)
        m.fit([train_c])
        raw = m.predict([eval_c])
        return raw[0]["score"] if isinstance(raw[0], dict) else raw[0]

    score_cal_dict: Dict[str, np.ndarray] = {}
    score_te_dict:  Dict[str, np.ndarray] = {}
    for c in clients:
        m     = GDN(**cfg)
        c_cal = dict(c, X_test=c["X_cal"], y_test=c["y_cal"])
        score_cal_dict[c["client_name"]] = _gdn_scores(m, c, c_cal)
        score_te_dict[c["client_name"]]  = _gdn_scores(m, c, c)

    auroc, auprc, f1, prec, rec, pred_dict = _calibrated_eval(
        clients, score_cal_dict, score_te_dict)
    per   = compute_per_client_metrics(clients, score_te_dict, pred_dict)
    macro = macro_from_per_client(per)

    print(f"  [GDN] AUROC={auroc:.4f} AUPRC={auprc:.4f} F1_cal={f1:.4f} "
          f"| macro-AUROC={macro['macro_auroc']:.4f}")

    overall = {
        "auroc": auroc, "auprc": auprc, "f1": f1,
        "precision": prec, "recall": rec,
        **macro,
        "label_fusion": False,
        "category": "Local-client",
    }
    return overall, per


                                                                           

def _not_implemented_stub(name: str):
    def _run(clients, dataset, seed, device):
        raise NotImplementedError(
            f"{name} not yet wired into run_fair_comparison.py. "
            f"Add its runner analogous to run_fl_stam / run_gdn.")
    return _run


run_fedanomaly = _not_implemented_stub("FedAnomaly")
run_pefad      = _not_implemented_stub("PeFAD")
run_ganf       = _not_implemented_stub("GANF")
run_mtad_gat   = _not_implemented_stub("MTAD-GAT")
run_tranad     = _not_implemented_stub("TranAD")


                                                                            

                                                                   
METHODS: Dict[str, callable] = {
    "fedhgf":    run_fedhgf,
    "fl_stam":   run_fl_stam,
    "gdn":       run_gdn,
    "fedanomaly":run_fedanomaly,
    "pefad":     run_pefad,
    "ganf":      run_ganf,
    "mtad_gat":  run_mtad_gat,
    "tranad":    run_tranad,
}

                                                         
MAIN_METHODS = ["fedhgf", "fl_stam", "gdn"]


                                                                            

def _run_anchor_ablation(args):
    """
    Anchor protocol ablation.
    WADI supports fixed physical anchors, all zone-3 anchors, and variance anchors.
    SWaT supports public FIT anchors and local-FIT-only anchors.
    """
    ds = args.dataset.split("_")[0]
    if ds == "wadi":
        variants = ["wadi", "wadi_all15", "wadi_variance"]
    elif ds == "swat":
        variants = ["swat", "swat_localfit"]
    else:
        print(f"  [anchor ablation] {ds} 暂无预设 anchor 变体，跑主协议")
        variants = [args.dataset]

    print(f"\n  [Anchor Ablation] variants={variants}")
    all_summary: Dict[str, dict] = {}
    seeds = [int(s) for s in args.seeds.split(",")]

    for var in variants:
        per_seed_results = []
        for seed in seeds:
            set_seed(seed)
            try:
                clients, n_anchor, anchor_cols = load_clients(var, seed, args.quick)
                overall, _ = run_fedhgf(
                    clients, n_anchor, var, seed,
                    fedhgf_full=False, device=args.device)
                per_seed_results.append(overall)
            except Exception as e:
                print(f"  [ERROR] {var} seed={seed}: {e}")
        if per_seed_results:
            keys = ["auroc", "auprc", "f1", "macro_auroc", "macro_f1"]
            all_summary[var] = {
                k: mean(r[k] for r in per_seed_results if k in r)
                for k in keys
            }

    print("\n  [Anchor Ablation 汇总]")
    print(f"  {'Variant':<22} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} "
          f"{'MacroAUROC':>11} {'MacroF1':>9}")
    for var, r in all_summary.items():
        print(f"  {var:<22} {r.get('auroc', float('nan')):>7.4f} "
              f"{r.get('auprc', float('nan')):>7.4f} {r.get('f1', float('nan')):>7.4f} "
              f"{r.get('macro_auroc', float('nan')):>11.4f} "
              f"{r.get('macro_f1', float('nan')):>9.4f}")

    out_dir = _this / f"results_{ds}_fair"
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_results(all_summary, str(out_dir), f"ablation_anchor_{ds}_{ts}")


def _run_label_ablation(args):
    """
    Label fusion 消融：
      FedHGF-unsup       (no calibration)
      FedHGF-threshold   (use cal labels only for threshold)
      FedHGF-Full        (use cal labels for fusion weight + threshold)
    """
    dataset = args.dataset
    seeds   = [int(s) for s in args.seeds.split(",")]
    spec    = get_fair_spec(dataset)

    variants = {
        "FedHGF (no-fusion, ratio-thr)": False,         
        "FedHGF-Full (label-fusion)":    True,              
    }
    all_summary: Dict[str, dict] = {}
    for label, use_full in variants.items():
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            try:
                clients, n_anchor, _ = load_clients(dataset, seed, args.quick)
                overall, _ = run_fedhgf(
                    clients, n_anchor, dataset, seed,
                    fedhgf_full=use_full, device=args.device)
                per_seed.append(overall)
            except Exception as e:
                print(f"  [ERROR] {label} seed={seed}: {e}")
        if per_seed:
            keys = ["auroc", "auprc", "f1", "macro_auroc"]
            all_summary[label] = {k: mean(r[k] for r in per_seed if k in r) for k in keys}

    print("\n  [Label Fusion Ablation 汇总]")
    for var, r in all_summary.items():
        print(f"  {var:<40} AUROC={r.get('auroc', float('nan')):.4f}  "
              f"MacroAUROC={r.get('macro_auroc', float('nan')):.4f}")

    out_dir = _this / spec.get("out_dir", f"results_{dataset}_fair")
    ts = time.strftime("%Y%m%d_%H%M%S")
    save_results(all_summary, str(out_dir), f"ablation_label_{dataset}_{ts}")


                                                                       

def run_experiment(args):
    dataset = args.dataset
    seeds   = [int(s) for s in args.seeds.split(",")]
    if args.method == "all":
        methods = MAIN_METHODS
    else:
        methods = [args.method]

    print("=" * 70)
    print(f"  Fair Benchmark | dataset={dataset} | methods={methods}")
    spec = get_fair_spec(dataset)
    tier = spec.get("tier", "?")
    print(f"  Tier={tier} | seeds={seeds} | device={args.device}")
    print("=" * 70)
    print(f"\n  [Anchor 设计]\n  {spec.get('anchor_design', 'N/A')}\n")

    all_results:     Dict[str, List[dict]] = {m: [] for m in methods}
    all_per_client:  Dict[str, List[list]] = {m: [] for m in methods}
    input_table_rows: List[dict] = []

                                                                                       
    _info_seeds = seeds[:1] if args.info_only else seeds

    for seed in _info_seeds if args.info_only else seeds:
        set_seed(seed)
        t0 = time.time()
        clients, n_anchor, anchor_cols = load_clients(
            dataset, seed, args.quick,
            label_mode=getattr(args, 'label_mode', None))
        print(f"\n  seed={seed}: {len(clients)} clients, "
              f"n_anchor={n_anchor}, anchor={anchor_cols}")
        print_feature_report(clients, "all methods (identical input)")

        if args.info_only:
            continue

        for method in methods:
            try:
                if method == "fedhgf":
                    _score_orient = getattr(args, 'score_orient', None)
                    _enc_type     = getattr(args, 'encoder_type', None)
                    _cfg_extra = {}
                    if _score_orient:
                        _cfg_extra["score_orient"] = _score_orient
                    if _enc_type:
                        _cfg_extra["encoder_type"] = _enc_type
                                                                                 
                    for _k in ("lambda_c", "lambda_v", "lambda_g", "lambda_t",
                               "tf_layers", "tf_ffn", "tf_dropout",
                               "patch_len", "patch_stride",
                               "flow_blocks", "flow_hidden", "flow_epochs",
                               "pred_rate_low", "pred_rate_high",
                               "fusion_search_beta",
                               "center_score_mode",
                               "batch_size", "lr", "flow_lr",
                               "n_rounds", "use_prediction_loss", "lambda_pred",
                               "w_fusion_per_client"):
                        _v = getattr(args, _k, None)
                        if _v is not None:
                            _cfg_extra[_k] = _v
                                                                   
                    _fc_extra = None
                    _fc_raw = getattr(args, 'fusion_candidates', None)
                    if _fc_raw:
                        _fc_extra = []
                        for trip in _fc_raw.split(";"):
                            parts = [float(x) for x in trip.strip().split(",")]
                            if len(parts) == 3:
                                _fc_extra.append(tuple(parts))
                    _track = getattr(args, 'track_convergence', False)
                    if _track:
                        if _cfg_extra is None:
                            _cfg_extra = {}
                        _cfg_extra['track_full_convergence'] = True
                    _ret = run_fedhgf(
                        clients, n_anchor, dataset, seed,
                        fedhgf_full=args.fedhgf_full, device=args.device,
                        w_fusion=args.w_fusion,
                        threshold_mode=getattr(args, 'threshold_mode', None),
                        target_anom_rate=getattr(args, 'target_anom_rate', None),
                        fusion_mode=getattr(args, 'fusion_mode', None),
                        cfg_extra=_cfg_extra if _cfg_extra else None,
                        fusion_candidates_extra=_fc_extra,
                        return_round_history=_track)
                    if _track:
                        overall, per, round_hist = _ret
                        if round_hist:
                            _rh_dir = _this / spec.get('out_dir', f'results_{dataset}_fair')
                            _rh_dir.mkdir(parents=True, exist_ok=True)
                            _rh_ts = time.strftime('%Y%m%d_%H%M%S')
                            _rh_path = _rh_dir / f'convergence_{dataset}_{_rh_ts}.csv'
                            import csv as _csv
                            with open(_rh_path, 'w', newline='', encoding='utf-8') as _fh:
                                _wr = _csv.DictWriter(_fh, fieldnames=['dataset','method','seed','round','auroc','f1','stage'])
                                _wr.writeheader()
                                for rh in round_hist:
                                    _wr.writerow({'dataset': dataset, 'method': 'FedHGF', 'seed': seed,
                                                  'round': rh.get('round',''), 'auroc': rh.get('auroc',''),
                                                  'f1': rh.get('f1',''), 'stage': rh.get('stage','')})
                            print(f'  [Convergence] 保存: {_rh_path}')
                    else:
                        overall, per = _ret
                    proto = "anchor+aux, own graph"
                    agg   = "encoder only"
                elif method == "fl_stam":
                    overall, per = run_fl_stam(clients, dataset, seed, args.device)
                    dims  = [c.get("n_k", c["X_train"].shape[2]) for c in clients]
                    proto = f"anchor+aux padded to max={max(dims)}"
                    agg   = "full backbone (FedAvg)"
                elif method == "gdn":
                    overall, per = run_gdn(clients, dataset, seed, args.device)
                    proto = "per-client anchor+aux (local)"
                    agg   = "none (local training)"
                else:
                    fn = METHODS.get(method)
                    if fn is None:
                        print(f"  [WARN] 未知方法: {method}，跳过")
                        continue
                    overall, per = fn(clients, dataset, seed, args.device)
                    proto = "N/A"
                    agg   = "N/A"

                all_results[method].append(overall)
                all_per_client[method].extend(
                    [{**r, "method": method, "seed": seed} for r in per])

                                          
                if seed == seeds[0]:
                    input_table_rows.append(input_transparency_row(
                        dataset, method, clients, proto, agg))

            except NotImplementedError as e:
                print(f"  [SKIP] {method}: {e}")
            except Exception as e:
                print(f"  [ERROR] {method} seed={seed}: {e}")
                import traceback; traceback.print_exc()

        print(f"  seed={seed} 完成，耗时 {time.time()-t0:.1f}s")

    if args.info_only:
        return

                                                                          
    summary: Dict[str, dict] = {}
    for method, results in all_results.items():
        if not results:
            continue
        pooled_keys = ["auroc", "auprc", "f1", "precision", "recall"]
        macro_keys  = ["macro_auroc", "macro_auprc", "macro_f1"]
        mean_vals = {}
        std_vals  = {}
        for k in pooled_keys + macro_keys:
            vals = [r[k] for r in results if k in r and not np.isnan(r[k])]
            if vals:
                mean_vals[k] = mean(vals)
                std_vals[k + "_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary[method] = {
            **mean_vals, **std_vals,
            "n_seeds": len(results),
            "label_fusion": results[0].get("label_fusion", False),
            "category": results[0].get("category", ""),
            "tier": tier,
        }

                                                                        
    print("\n" + "=" * 78)
    print(f"  Fair Comparison Summary — {dataset.upper()} (Tier {tier})")
    print("=" * 78)
    header = (f"  {'Category':<14} {'Method':<18} {'AUROC':>7} ±   "
              f"{'AUPRC':>7} {'F1':>7}  {'MacroAUROC':>11} {'MacroF1':>9}")
    print(header)
    print("  " + "-" * 74)
    for method, r in summary.items():
        cat  = r.get("category", "")
        name = ("FedHGF-Full" if r["label_fusion"] else
                "FedHGF" if method == "fedhgf" else method.upper())
        auroc_s = f"{r.get('auroc', float('nan')):.4f}"
        auroc_std_s = f"{r.get('auroc_std', 0):.4f}"
        print(f"  {cat:<14} {name:<18} {auroc_s:>7}±{auroc_std_s:<6} "
              f"{r.get('auprc', float('nan')):>7.4f} "
              f"{r.get('f1', float('nan')):>7.4f}  "
              f"{r.get('macro_auroc', float('nan')):>11.4f} "
              f"{r.get('macro_f1', float('nan')):>9.4f}")

                                                                         
    out_dir = _this / spec.get("out_dir", f"results_{dataset}_fair")
    ts = time.strftime("%Y%m%d_%H%M%S")

    save_results(summary, str(out_dir), f"fair_{dataset}_{ts}")

    if all_per_client:
        rows = [r for rows in all_per_client.values() for r in rows]
        save_csv(rows, out_dir / f"per_client_{dataset}_{ts}.csv")

    if input_table_rows:
        save_csv(input_table_rows, out_dir / f"input_table_{dataset}_{ts}.csv")
        print("\n  [输入透明表预览]")
        print(f"  {'Dataset':<10} {'Method':<14} {'Dims':<18} "
              f"{'Protocol':<40} {'Agg'}")
        for row in input_table_rows:
            print(f"  {row['dataset']:<10} {row['method']:<14} "
                  f"{row['client_dims']:<18} {row['input_protocol']:<40} "
                  f"{row['aggregated_params']}")


def main():
    ap = argparse.ArgumentParser(
        description="Fair Benchmark — FedHGF vs Baselines (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", default=None,
                    choices=sorted(DATASET_REGISTRY_FAIR.keys()),
                    help="Dataset key (use --list-datasets to see all)")
    ap.add_argument("--method", default="fedhgf",
                    choices=["all"] + list(METHODS.keys()))
    ap.add_argument("--seeds", default="42,123,2024",
                    help="Comma-separated random seeds (default: 3 seeds)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true",
                    help="Cap train rows for fast smoke test")
    ap.add_argument("--fedhgf-full", dest="fedhgf_full", action="store_true",
                    help="FedHGF-Full: enable label-assisted fusion (ablation only)")
    ap.add_argument("--info-only", dest="info_only", action="store_true",
                    help="Print dataset info and exit (no training)")
    ap.add_argument("--ablation", default=None,
                    choices=["anchor", "label", "input"],
                    help="Run a specific ablation suite instead of main experiment")
    ap.add_argument("--list-datasets", dest="list_datasets", action="store_true",
                    help="Print all datasets by tier and exit")
    ap.add_argument("--w-fusion", dest="w_fusion", type=str, default=None,
                    help="Override fusion weights, e.g. 0.05,0.70,0.25")
    ap.add_argument("--w-fusion-per-client", dest="w_fusion_per_client", type=str, default=None,
                    help="Per-client fusion weights, semicolon-separated, e.g. '0.7,0.2,0.1;0.5,0.3,0.2'")
    ap.add_argument("--threshold-mode", dest="threshold_mode", type=str, default=None,
                    choices=["f1", "f1_guard", "f1_fpr_guard", "f1_rate_guard", "rate", "ratio",
                             "quantile", "normal_percentile"],
                    help="Override adaptive_threshold_mode (e.g. rate)")
    ap.add_argument("--target-anom-rate", dest="target_anom_rate", type=float, default=None,
                    help="Override target_anom_rate (e.g. 0.10 for rate mode)")
    ap.add_argument("--fusion-mode", dest="fusion_mode", type=str, default=None,
                    choices=["fixed", "calib_small", "calib_small_balanced"],
                    help="Override fusion_mode: fixed=global weights, calib_small=6-cand, calib_small_balanced=11-cand symmetric")
    ap.add_argument("--fusion-candidates", dest="fusion_candidates", type=str, default=None,
                    help="Extra fusion weight candidates for calib modes, e.g. "
                         "'0.25,0.35,0.40;0.20,0.40,0.40;0.30,0.30,0.40' "
                         "(semicolon-separated triples appended to default list)")
    ap.add_argument("--pred-rate-low", dest="pred_rate_low", type=float, default=None,
                    help="Lower bound of pred-rate window for f1_guard mode")
    ap.add_argument("--pred-rate-high", dest="pred_rate_high", type=float, default=None,
                    help="Upper bound of pred-rate window for f1_guard mode")
    ap.add_argument("--fusion-beta", dest="fusion_search_beta", type=float, default=None,
                    help="F-beta for fusion search: 1.0=F1 (default), 0.5=precision-heavy F0.5")
    ap.add_argument("--center-score-mode", dest="center_score_mode", type=str, default=None,
                    choices=["global", "local", "hybrid"],
                    help="Override center score mode: global, local, or hybrid")
    ap.add_argument("--score-orient", dest="score_orient", type=str, default=None,
                    choices=["none", "calib_auc"],
                    help="Score orientation: calib_auc flips branches with cal-AUROC<0.5")
    ap.add_argument("--encoder-type", dest="encoder_type", type=str, default=None,
                    choices=["npformer_gp", "gru"],
                    help="Stage I encoder. npformer_gp=default, gru=fallback")
                                                                          
    ap.add_argument("--lambda-c", dest="lambda_c", type=float, default=None,
                    help="Override lambda_c (center loss weight)")
    ap.add_argument("--lambda-v", dest="lambda_v", type=float, default=None,
                    help="Override lambda_v (variance floor weight)")
    ap.add_argument("--lambda-g", dest="lambda_g", type=float, default=None,
                    help="Override lambda_g (graph smoothing weight)")
    ap.add_argument("--lambda-t", dest="lambda_t", type=float, default=None,
                    help="Override lambda_t (temporal smoothing weight)")
                                                                          
    ap.add_argument("--tf-layers", dest="tf_layers", type=int, default=None,
                    help="Override NPFormer-GP transformer layers (default 2)")
    ap.add_argument("--tf-ffn", dest="tf_ffn", type=int, default=None,
                    help="Override NPFormer-GP transformer ffn dim (default 128)")
    ap.add_argument("--tf-dropout", dest="tf_dropout", type=float, default=None,
                    help="Override NPFormer-GP transformer dropout (default 0.1)")
    ap.add_argument("--patch-len", dest="patch_len", type=int, default=None,
                    help="Override NPFormer-GP patch_len (default 4)")
    ap.add_argument("--patch-stride", dest="patch_stride", type=int, default=None,
                    help="Override NPFormer-GP patch_stride (default 2)")
                                                                           
    ap.add_argument("--flow-blocks", dest="flow_blocks", type=int, default=None,
                    help="Override Stage-II MAF flow_blocks (default 4)")
    ap.add_argument("--flow-hidden", dest="flow_hidden", type=int, default=None,
                    help="Override Stage-II MAF flow_hidden (default 48)")
    ap.add_argument("--flow-epochs", dest="flow_epochs", type=int, default=None,
                    help="Override Stage-II MAF flow_epochs (default 15)")
                                                                         
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                    help="Override batch_size (default from BASE_CFG_FAIR)")
    ap.add_argument("--lr", dest="lr", type=float, default=None,
                    help="Override Stage-I learning rate")
    ap.add_argument("--flow-lr", dest="flow_lr", type=float, default=None,
                    help="Override Stage-II flow learning rate")
    ap.add_argument("--n-rounds", dest="n_rounds", type=int, default=None,
                    help="Override number of federated training rounds (default 10)")
    ap.add_argument("--prediction-loss", dest="use_prediction_loss", action="store_true", default=None,
                    help="Enable prediction loss (GRU encoder only)")
    ap.add_argument("--lambda-pred", dest="lambda_pred", type=float, default=None,
                    help="Prediction loss weight (default 1.0)")
                                                                         
    ap.add_argument("--label-mode", dest="label_mode", type=str, default=None,
                    choices=["any", "last", "center", "majority"],
                    help="Window label mode: any=max-in-window (default), last=last-timestamp, center=center-timestamp, majority=>50%%")
    ap.add_argument("--track-convergence", dest="track_convergence", action="store_true", default=False,
                    help="Track per-round full-model AUROC for convergence plot (FedHGF only)")
    args = ap.parse_args()
    if args.w_fusion is not None:
        args.w_fusion = tuple(float(x) for x in args.w_fusion.split(","))
    if args.w_fusion_per_client is not None:
        raw = args.w_fusion_per_client.strip()
        if "(" in raw:
            import re
            parts = re.findall(r"\(([^)]+)\)", raw)
        else:
            parts = raw.split(";")
        args.w_fusion_per_client = [
            tuple(float(x) for x in part.split(","))
            for part in parts]

    if args.list_datasets:
        print("\nFair Benchmark - Datasets by Tier")
        list_datasets_by_tier()
        print()
        return

    if args.dataset is None:
        ap.error("--dataset is required unless --list-datasets is used")

    if args.ablation == "anchor":
        _run_anchor_ablation(args)
    elif args.ablation == "label":
        _run_label_ablation(args)
    elif args.ablation == "input":
                                                                     
        orig = args.dataset
        for ds in [orig, orig + "_localfit"]:
            if ds in DATASET_REGISTRY_FAIR:
                args.dataset = ds
                run_experiment(args)
        args.dataset = orig
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
