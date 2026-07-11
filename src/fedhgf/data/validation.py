"""Protocol validation checks."""

from __future__ import annotations

from collections import defaultdict

from .schema import FederationDataset


def validate_temporal_disjointness(federation: FederationDataset) -> None:
    for client in federation.clients:
        train = {i for w in client.train_index for i in range(w.raw_start, w.raw_end)}
        cal = {i for w in client.calibration_index for i in range(w.raw_start, w.raw_end)}
        if train & cal:
            raise ValueError(f"{client.client_id}: train/calibration windows overlap")


def validate_anchor_semantics(federation: FederationDataset) -> None:
    if federation.shared_anchor_observations:
        return
    by_semantic: dict[str, list[str]] = defaultdict(list)
    for client in federation.clients:
        for node in client.client_spec.anchor_nodes:
            by_semantic[node.semantic_id].append(node.source_series_id)
    for semantic_id, source_ids in by_semantic.items():
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(
                f"independent-site protocol reuses source_series_id for {semantic_id}"
            )


def validate_model_features_are_label_free(federation: FederationDataset) -> None:
    for client in federation.clients:
        if hasattr(client, "test_y") or hasattr(client, "calibration_y"):
            raise ValueError(f"{client.client_id}: ClientFeatures must not contain labels")


def validate_protocol(federation: FederationDataset) -> None:
    validate_temporal_disjointness(federation)
    validate_anchor_semantics(federation)
    validate_model_features_are_label_free(federation)
