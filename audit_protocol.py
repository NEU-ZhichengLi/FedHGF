from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from run_fair_comparison import load_clients


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _window_rows(client: dict, split_name: str) -> Iterable[dict]:
    audit = client.get("split_audit", {})
    info = audit.get(split_name, {})
    window_len = int(audit.get("window_len", 0))
    stride = int(audit.get("stride", 1))
    start0 = int(info.get("row_start", 0))
    n_rows = int(info.get("n_rows", 0))
    source = info.get("source", "")
    labels = client.get(f"y_{split_name}")
    if split_name == "test":
        labels = client.get("y_test")
    elif split_name == "cal":
        labels = client.get("y_cal")
    elif split_name == "train":
        labels = client.get("y_train")
    n_windows = max(0, (n_rows - window_len) // stride + 1) if window_len else 0
    for i in range(n_windows):
        ws = start0 + i * stride
        label = int(labels[i]) if labels is not None and i < len(labels) else ""
        yield {
            "dataset": audit.get("dataset", ""),
            "client": client.get("client_name", ""),
            "split": split_name,
            "source": json.dumps(source, ensure_ascii=False),
            "window_index": i,
            "row_start": ws,
            "row_end_exclusive": ws + window_len,
            "label_for_final_metrics": label,
        }


def _anchor_audit_rows(clients: list[dict]) -> list[dict]:
    rows = []
    if len(clients) < 2:
        return rows
    ref = clients[0]
    n_anchor = len(ref.get("anchor_names", []))
    for split in ("X_train", "X_cal", "X_test"):
        ref_x = ref[split][:, :, :n_anchor, :]
        for other in clients[1:]:
            other_x = other[split][:, :, :n_anchor, :]
            comparable = ref_x.shape == other_x.shape
            if comparable:
                max_abs = float(np.max(np.abs(ref_x - other_x))) if ref_x.size else 0.0
                mean_abs = float(np.mean(np.abs(ref_x - other_x))) if ref_x.size else 0.0
            else:
                max_abs = float("nan")
                mean_abs = float("nan")
            rows.append({
                "reference_client": ref.get("client_name", ""),
                "other_client": other.get("client_name", ""),
                "split": split.replace("X_", ""),
                "anchor_names_reference": json.dumps(ref.get("anchor_names", []), ensure_ascii=False),
                "anchor_names_other": json.dumps(other.get("anchor_names", []), ensure_ascii=False),
                "same_shape": comparable,
                "max_abs_diff_after_train_normalization": max_abs,
                "mean_abs_diff_after_train_normalization": mean_abs,
                "identical_after_train_normalization": comparable and max_abs == 0.0,
                "interpretation": (
                    "replicated public-state/context anchor"
                    if comparable and max_abs == 0.0
                    else "not an identical replicated sequence"
                ),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export split and anchor audit evidence for the FedHGF protocol.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--label-mode", default=None,
                    choices=["any", "last", "center", "majority"])
    ap.add_argument("--out-dir", default="protocol_audit")
    args = ap.parse_args()

    clients, n_anchor, anchor_cols = load_clients(
        args.dataset, args.seed, args.quick, label_mode=args.label_mode)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_seed{args.seed}"

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_clients": len(clients),
        "n_anchor": n_anchor,
        "anchor_cols": anchor_cols,
        "clients": [],
    }
    for c in clients:
        summary["clients"].append({
            "client_name": c.get("client_name"),
            "n_k": c.get("n_k"),
            "anchor_names": c.get("anchor_names", []),
            "aux_names": c.get("aux_names", []),
            "split_protocol": c.get("split_protocol"),
            "cal_anomaly_windows": int(np.asarray(c.get("y_cal", [])).sum()),
            "test_windows": int(len(c.get("y_test", []))),
            "test_anomaly_windows": int(np.asarray(c.get("y_test", [])).sum()),
            "test_anomaly_rate": float(np.asarray(c.get("y_test", [])).mean())
                                 if len(c.get("y_test", [])) else 0.0,
            "split_audit": c.get("split_audit", {}),
        })
    (out / f"{stem}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8")

    window_path = out / f"{stem}_windows.csv"
    with window_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset", "client", "split", "source", "window_index",
            "row_start", "row_end_exclusive", "label_for_final_metrics",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in clients:
            for split in ("train", "cal", "test"):
                writer.writerows(_window_rows(c, split))

    anchor_path = out / f"{stem}_anchor_identity.csv"
    with anchor_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "reference_client", "other_client", "split",
            "anchor_names_reference", "anchor_names_other", "same_shape",
            "max_abs_diff_after_train_normalization",
            "mean_abs_diff_after_train_normalization",
            "identical_after_train_normalization", "interpretation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_anchor_audit_rows(clients))

    print(f"wrote {out / f'{stem}_summary.json'}")
    print(f"wrote {window_path}")
    print(f"wrote {anchor_path}")


if __name__ == "__main__":
    main()
