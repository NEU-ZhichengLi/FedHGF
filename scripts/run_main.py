"""Canonical main runner entry point.

This entry point builds the v2 protocol first, then adapts it to the existing
FedGAD model implementation. Evaluation labels stay outside the model-facing
ClientFeatures object.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset_registry_fair import BASE_CFG_FAIR
from fedgad_full import FedGAD
from run_fair_comparison import compute_per_client_metrics, macro_from_per_client
from src.fedhgf.data.protocol_builder import (
    build_hai_shared_context_protocol,
    to_legacy_clients,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hai"], default="hai")
    ap.add_argument("--data-dir", default=str(Path("Data") / "HAI 21.03"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    federation = build_hai_shared_context_protocol(args.data_dir)
    clients, n_anchor, _ = to_legacy_clients(federation)

    cfg = dict(BASE_CFG_FAIR)
    cfg.update({
        "device": args.device,
        "seed": args.seed,
        "adaptive_threshold_mode": "quantile",
        "fusion_mode": "fixed",
        "use_label_assisted_fusion": False,
        "w_fusion": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    })
    model = FedGAD(n_anchor=n_anchor, **cfg)
    model.fit(clients)
    raw = model.predict(clients)
    score_dict = {r["client_name"]: r["score"] for r in raw}
    pred_dict = {r["client_name"]: r["y_pred"] for r in raw}
    tau_dict = {r["client_name"]: r.get("tau", float("nan")) for r in raw}
    per = compute_per_client_metrics(clients, score_dict, pred_dict, tau_dict)
    macro = macro_from_per_client(per)
    print({"macro": macro, "per_client": per})


if __name__ == "__main__":
    main()
