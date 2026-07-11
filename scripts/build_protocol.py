"""Build and validate a canonical protocol instance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fedhgf.data.protocol_builder import build_hai_shared_context_protocol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hai"], default="hai")
    ap.add_argument("--data-dir", default=str(Path("Data") / "HAI 21.03"))
    ap.add_argument("--window-length", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--train-fraction", type=float, default=0.80)
    ap.add_argument("--guard-gap", type=int, default=15)
    args = ap.parse_args()

    federation = build_hai_shared_context_protocol(
        args.data_dir,
        window_length=args.window_length,
        stride=args.stride,
        train_fraction=args.train_fraction,
        guard_gap=args.guard_gap,
    )
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
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
