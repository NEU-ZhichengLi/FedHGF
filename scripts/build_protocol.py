"""Build and validate a canonical protocol instance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fedhgf.data.protocol_builder import build_protocol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hai", "wadi", "swat", "batadal"], default="hai")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--window-length", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--train-fraction", type=float, default=0.80)
    ap.add_argument("--guard-gap", type=int, default=15)
    args = ap.parse_args()

    data_dir = args.data_dir or str(Path("Data") / {
        "hai": "HAI 21.03",
        "wadi": "WADI",
        "swat": "SWAT",
        "batadal": "BATADAL",
    }[args.dataset])
    kwargs = {
        "window_length": args.window_length,
        "stride": args.stride,
        "train_fraction": args.train_fraction,
        "guard_gap": args.guard_gap,
    }
    if args.dataset == "batadal" and args.window_length == 16 and args.stride == 4:
        kwargs.update({"window_length": 32, "stride": 1, "guard_gap": 0})
    federation = build_protocol(args.dataset, data_dir, **kwargs)
    payload = {
        "dataset": federation.dataset,
        "protocol_version": federation.protocol_version,
        "federation_type": federation.federation_type,
        "shared_anchor_observations": federation.shared_anchor_observations,
        "n_clients": len(federation.clients),
        "clients": [c.client_id for c in federation.clients],
        "n_anchor": federation.n_anchor,
        "train_rows": federation.temporal_split.train.n_rows,
        "calibration_rows": federation.temporal_split.calibration.n_rows,
        "test_rows": federation.temporal_split.test.n_rows,
        "test_windows": {
            client.client_id: len(client.test_x)
            for client in federation.clients
        },
        "client_windows": {
            client.client_id: {
                "train": len(client.train_x),
                "calibration": len(client.calibration_x),
                "test": len(client.test_x),
            }
            for client in federation.clients
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
