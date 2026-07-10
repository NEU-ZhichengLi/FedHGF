from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                              precision_score, recall_score, roc_auc_score)


def evaluate(results: List[dict]) -> dict:
    y_true_all = np.concatenate([r["y_true"] for r in results])
    y_pred_all = np.concatenate([r["y_pred"] for r in results])
    score_all  = np.concatenate([r["score"]  for r in results])

    per_client = []
    for r in results:
        yt, yp, sc = r["y_true"], r["y_pred"], r["score"]
        n_anom = int(yt.sum())
        f1_val = f1_score(yt, yp, zero_division=0)

                                
        if n_anom == 0:
            note = "⚠ no_anom"                           
        elif n_anom < 30:
            note = f"⚠ sparse({n_anom})"                 
        elif f1_val == 0.0:
            note = "⚠ F1=0"                             
        else:
            note = ""

        row = {
            "client":    r.get("client_name", str(r.get("client_id", "client"))),
            "n_test":    int(len(yt)),
            "n_anom":    n_anom,
            "f1":        f1_val,
            "precision": precision_score(yt, yp, zero_division=0),
            "recall":    recall_score(yt, yp, zero_division=0),
            "note":      note,                        
        }
        if len(np.unique(yt)) >= 2:
            row["auroc"] = roc_auc_score(yt, sc)
            row["auprc"] = average_precision_score(yt, sc)
        else:
            row["auroc"] = float("nan")
            row["auprc"] = float("nan")
        per_client.append(row)

    overall = {
        "f1":        f1_score(y_true_all, y_pred_all, zero_division=0),
        "precision": precision_score(y_true_all, y_pred_all, zero_division=0),
        "recall":    recall_score(y_true_all, y_pred_all, zero_division=0),
        "worst_f1":  min(pc["f1"] for pc in per_client),
        "macro_f1":  float(np.mean([pc["f1"] for pc in per_client])),
        "note":      "",                   
    }
    if len(np.unique(y_true_all)) >= 2:
        overall["auroc"] = roc_auc_score(y_true_all, score_all)
        overall["auprc"] = average_precision_score(y_true_all, score_all)
    else:
        overall["auroc"] = float("nan")
        overall["auprc"] = float("nan")

    return {"overall": overall, "per_client": per_client}


def _fmt(value) -> str:
    if value is None:
        return f"{'-':>8}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{str(value):>8}"
    if math.isnan(v):
        return f"{'-':>8}"
    return f"{v:>8.4f}"


def print_table(title: str, rows: List[dict]) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Method':<28} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} "
          f"{'mF1':>8} {'wF1':>8}  Note")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 14}")
    for r in rows:
        note = r.get("note", "")
        print(
            f"  {r['name']:<28} "
            f"{_fmt(r.get('auroc'))} "
            f"{_fmt(r.get('auprc'))} "
            f"{_fmt(r.get('f1'))} "
            f"{_fmt(r.get('macro_f1'))} "
            f"{_fmt(r.get('worst_f1'))}"
            + (f"  {note}" if note else "")
        )
    print(f"{'=' * 90}")


def save_result_artifacts(result: dict, out_dir: str, stem: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall_path     = out / f"overall_{stem}.csv"
    per_client_path  = out / f"per_client_{stem}.csv"
    json_path        = out / f"results_{stem}.json"

    pd.DataFrame([result["overall"]]).to_csv(overall_path, index=False)
    pd.DataFrame(result["per_client"]).to_csv(per_client_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return {
        "overall_csv":    str(overall_path),
        "per_client_csv": str(per_client_path),
        "json":           str(json_path),
    }
