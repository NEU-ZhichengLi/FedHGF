"""Canonical FedHGF main runner.

The model receives label-free client feature dictionaries and returns scores
only. Calibration and evaluation are performed outside the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedgad_full import FedGAD
from src.fedhgf.calibration import QuantileCalibrator
from src.fedhgf.data.protocol_builder import (
    build_hai_shared_context_protocol,
    to_model_clients,
)
from src.fedhgf.evaluation.window_metrics import window_metrics


def _base_model_config(device: str, seed: int) -> dict:
    return {
        "n_rounds": 10,
        "local_epochs": 1,
        "d_h": 64,
        "flow_blocks": 4,
        "flow_hidden": 128,
        "batch_size": 128,
        "lr": 3e-4,
        "flow_lr": 3e-4,
        "lambda_c": 1.0,
        "lambda_g": 0.005,
        "lambda_t": 0.005,
        "lambda_v": 0.2,
        "gamma_var": 0.05,
        "C_g": 1.0,
        "sigma_g": 1.0,
        "C_theta": 1.0,
        "sigma_theta": 1.0,
        "C_c": 1.0,
        "sigma_c": 1.0,
        "use_dp": False,
        "use_calibration": True,
        "use_prediction_loss": False,
        "center_score_mode": "local",
        "score_mode": "both",
        "use_flow": True,
        "collapse_std_thr": 0.001,
        "hybrid_center_alpha": 0.0,
        "adaptive_hybrid_alpha": False,
        "w_fusion": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "use_data_driven_cross_block": True,
        "graph_residual_mode": "node_topk_q90",
        "relation_value_weight": 0.5,
        "encoder_type": "npformer_gp",
        "patch_len": 4,
        "patch_stride": 2,
        "tf_layers": 2,
        "tf_heads": 4,
        "tf_ffn": 128,
        "tf_dropout": 0.1,
        "device": device,
        "seed": seed,
    }


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path: str | Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line!r}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _apply_privacy_config(model_config: dict, experiment_config: dict) -> dict:
    privacy = experiment_config.get("privacy", {})
    enabled = bool(privacy.get("enabled", False))
    model_config["use_dp"] = enabled

    server_visible = privacy.get("server_visible_channels", {})
    graph = server_visible.get("graph", {})
    encoder = server_visible.get("encoder", {})
    center = server_visible.get("center", {})
    if graph:
        model_config["C_g"] = float(graph.get("clip_norm", model_config["C_g"]))
        model_config["sigma_g"] = float(graph.get("noise_multiplier", model_config["sigma_g"]))
    if encoder:
        model_config["C_theta"] = float(encoder.get("clip_norm", model_config["C_theta"]))
        model_config["sigma_theta"] = float(encoder.get("noise_multiplier", model_config["sigma_theta"]))
    if center:
        model_config["C_c"] = float(center.get("clip_norm", model_config["C_c"]))
        model_config["sigma_c"] = float(center.get("noise_multiplier", model_config["sigma_c"]))

    return {
        "enabled": enabled,
        "dp_backend": "dp_simulator" if enabled else "none",
        "secure_aggregation_backend": privacy.get("secure_aggregation_backend", "assumed"),
        "delta": privacy.get("delta"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hai"], default="hai")
    ap.add_argument("--data-dir", default=str(Path("Data") / "HAI 21.03"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alert-budget", type=float, default=0.05)
    ap.add_argument("--experiment-config", default=str(Path("configs") / "experiments" / "main_no_dp.yaml"))
    args = ap.parse_args()

    experiment_config = _load_simple_yaml(args.experiment_config)
    federation = build_hai_shared_context_protocol(args.data_dir)
    clients, n_anchor, _ = to_model_clients(federation)

    model_config = _base_model_config(args.device, args.seed)
    privacy_report = _apply_privacy_config(model_config, experiment_config)
    model = FedGAD(n_anchor=n_anchor, **model_config)
    model.fit(clients)
    cal_scores = {r["client_name"]: r["score"] for r in model.score(clients, split="calibration")}
    test_scores = {r["client_name"]: r["score"] for r in model.score(clients, split="test")}

    per_client = []
    for client in federation.clients:
        name = client.client_id
        calibrator = QuantileCalibrator(alert_budget=args.alert_budget)
        calibrator.fit(cal_scores[name])
        pred = calibrator.predict(test_scores[name])
        y_true = federation.labels[name].test_y
        row = window_metrics(y_true, test_scores[name], pred)
        row.update({"client": name, "threshold": calibrator.threshold_})
        per_client.append(row)

    macro = {
        key: float(np.nanmean([row[key] for row in per_client]))
        for key in ("auroc", "auprc", "precision", "recall", "f1")
    }
    print(json.dumps({"privacy": privacy_report, "macro": macro, "per_client": per_client}, indent=2))


if __name__ == "__main__":
    main()
