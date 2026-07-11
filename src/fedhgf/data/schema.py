"""Typed containers for the FedHGF data protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RawDataset:
    normal_values: np.ndarray
    normal_timestamps: np.ndarray
    test_values: np.ndarray
    test_timestamps: np.ndarray
    feature_names: tuple[str, ...]
    test_value_parts: tuple[np.ndarray, ...] = ()
    test_timestamp_parts: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True, order=True)
class TemporalRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("TemporalRange.start must be non-negative")
        if self.end < self.start:
            raise ValueError("TemporalRange.end must be >= start")

    @property
    def n_rows(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TemporalSplit:
    train: TemporalRange
    calibration: TemporalRange
    test: TemporalRange


@dataclass(frozen=True)
class WindowIndex:
    raw_start: int
    raw_end: int
    start_timestamp: object
    end_timestamp: object


@dataclass(frozen=True)
class WindowedSegment:
    values: np.ndarray
    indices: tuple[WindowIndex, ...]


@dataclass(frozen=True)
class NodeSpec:
    local_node_id: str
    semantic_id: str
    source_series_id: str
    role: str

    def __post_init__(self) -> None:
        if self.role not in {"anchor", "auxiliary"}:
            raise ValueError("NodeSpec.role must be 'anchor' or 'auxiliary'")


@dataclass(frozen=True)
class ClientSpec:
    client_id: str
    site_id: str
    node_specs: tuple[NodeSpec, ...]

    @property
    def anchor_nodes(self) -> tuple[NodeSpec, ...]:
        return tuple(n for n in self.node_specs if n.role == "anchor")

    @property
    def auxiliary_nodes(self) -> tuple[NodeSpec, ...]:
        return tuple(n for n in self.node_specs if n.role == "auxiliary")


@dataclass(frozen=True)
class ClientFeatures:
    client_id: str
    feature_names: tuple[str, ...]
    anchor_names: tuple[str, ...]
    aux_names: tuple[str, ...]
    train_x: np.ndarray
    calibration_x: np.ndarray
    test_x: np.ndarray
    train_index: tuple[WindowIndex, ...]
    calibration_index: tuple[WindowIndex, ...]
    test_index: tuple[WindowIndex, ...]
    client_spec: ClientSpec
    graph_metadata: Mapping[str, object]


@dataclass(frozen=True)
class FederationDataset:
    dataset: str
    protocol_version: int
    federation_type: str
    shared_anchor_observations: bool
    clients: tuple[ClientFeatures, ...]
    n_anchor: int
    anchor_names: tuple[str, ...]
    temporal_split: TemporalSplit
    window_length: int
    stride: int
