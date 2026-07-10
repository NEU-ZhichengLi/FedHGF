"""Window-level anomaly metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def window_metrics(y_true: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.int64)
    n = min(len(y_true), len(scores), len(pred))
    y_true, scores, pred = y_true[:n], scores[:n], pred[:n]
    has_both = len(np.unique(y_true)) == 2
    return {
        "n_test": n,
        "n_anomaly": int(y_true.sum()),
        "natural_anomaly_rate": float(y_true.mean()) if n else 0.0,
        "auroc": float(roc_auc_score(y_true, scores)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, scores)) if has_both else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

