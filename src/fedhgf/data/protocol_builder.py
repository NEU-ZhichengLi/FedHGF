"""Canonical FedHGF protocol builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .normalization import TrainOnlyStandardizer
from .readers.hai import read_hai_raw
from .schema import (
    ClientFeatures,
    ClientSpec,
    FederationDataset,
    NodeSpec,
    WindowedSegment,
)
from .temporal_split import attach_test_range, split_normal_train_cal
from .validation import validate_protocol
from .windowing import windowize_segment

HAI_CLIENT_PROCESSES = ("P1", "P2", "P4")
HAI_ANCHOR_PROCESS = "P3"

WADI_ANCHORS = (
    "3_LT_001_PV",
    "3_FIT_001_PV",
    "3_AIT_001_PV",
    "3_AIT_002_PV",
    "3_AIT_003_PV",
    "3_AIT_004_PV",
    "3_AIT_005_PV",
    "3_P_001_STATUS",
)

SWAT_ANCHORS = ("FIT101", "FIT201", "FIT301", "FIT401", "FIT501", "FIT601")
SWAT_CLIENTS = {
    "Stage1": ("FIT101", "LIT101", "MV101", "P101", "P102"),
    "Stage2": ("FIT201", "AIT201", "AIT202", "AIT203", "MV201", "P201", "P202", "P203", "P204", "P205", "P206"),
    "Stage3": ("FIT301", "DPIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302"),
    "Stage4": ("FIT401", "AIT401", "AIT402", "LIT401", "P401", "P402", "P403", "P404", "UV401"),
    "Stage5": ("FIT501", "AIT501", "AIT502", "AIT503", "AIT504", "FIT502", "FIT503", "FIT504", "P501", "P502", "PIT501", "PIT502", "PIT503"),
    "Stage6": ("FIT601", "P601", "P602", "P603"),
}

BATADAL_ANCHORS = ("L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7")
BATADAL_CLIENTS = {
    "ZoneA": ("F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4", "F_PU5", "S_PU5", "F_PU6", "S_PU6"),
    "ZoneB": ("F_PU7", "S_PU7", "F_PU8", "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11", "F_V2", "S_V2"),
}


def _process_columns(feature_names: tuple[str, ...], process: str) -> tuple[str, ...]:
    return tuple(c for c in feature_names if c.startswith(f"{process}_"))


def _select_feature_block(
    values: np.ndarray,
    all_names: tuple[str, ...],
    selected: tuple[str, ...],
) -> np.ndarray:
    idx = [all_names.index(c) for c in selected]
    return values[:, idx, np.newaxis].astype(np.float32)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _numeric_frame(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    out = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return out.ffill().bfill().fillna(0.0).astype(np.float32).to_numpy()


def _concat_windowed(parts) -> tuple[np.ndarray, tuple]:
    values = [p.values for p in parts if len(p.values)]
    indices = tuple(i for p in parts for i in p.indices)
    if values:
        return np.concatenate(values, axis=0), indices
    first = parts[0].values
    return first[:0], indices


def _build_client(
    *,
    dataset: str,
    site_id: str,
    client_id: str,
    feature_names: tuple[str, ...],
    anchor_names: tuple[str, ...],
    aux_names: tuple[str, ...],
    train_values: np.ndarray,
    train_timestamps: np.ndarray,
    train_offset: int,
    cal_values: np.ndarray,
    cal_timestamps: np.ndarray,
    cal_offset: int,
    test_values: np.ndarray,
    test_timestamps: np.ndarray,
    window_length: int,
    stride: int,
) -> ClientFeatures:
    train_seg = windowize_segment(train_values, train_timestamps, window_length, stride, raw_offset=train_offset)
    cal_seg = windowize_segment(cal_values, cal_timestamps, window_length, stride, raw_offset=cal_offset)
    test_seg = windowize_segment(test_values, test_timestamps, window_length, stride, raw_offset=0)
    scaler = TrainOnlyStandardizer.fit(train_seg.values)
    node_specs = tuple(
        NodeSpec(name, name, f"{dataset}:{name}", "anchor")
        for name in anchor_names
    ) + tuple(
        NodeSpec(name, name, f"{dataset}:{client_id}:{name}", "auxiliary")
        for name in aux_names
    )
    return ClientFeatures(
        client_id=client_id,
        feature_names=feature_names,
        anchor_names=anchor_names,
        aux_names=aux_names,
        train_x=scaler.transform(train_seg.values),
        calibration_x=scaler.transform(cal_seg.values),
        test_x=scaler.transform(test_seg.values),
        train_index=train_seg.indices,
        calibration_index=cal_seg.indices,
        test_index=test_seg.indices,
        client_spec=ClientSpec(client_id=client_id, site_id=site_id, node_specs=node_specs),
        graph_metadata={
            "federation_type": "shared_context_vertical",
            "shared_anchor_observations": True,
            "normalization": scaler.fit_scope,
        },
    )


def _federation(
    *,
    dataset: str,
    clients: list[ClientFeatures],
    n_anchor: int,
    anchor_names: tuple[str, ...],
    split,
    window_length: int,
    stride: int,
) -> FederationDataset:
    fed = FederationDataset(
        dataset=dataset,
        protocol_version=2,
        federation_type="shared_context_vertical",
        shared_anchor_observations=True,
        clients=tuple(clients),
        n_anchor=n_anchor,
        anchor_names=anchor_names,
        temporal_split=split,
        window_length=window_length,
        stride=stride,
    )
    validate_protocol(fed)
    return fed


def build_hai_shared_context_protocol(
    data_dir: str | Path,
    *,
    window_length: int = 16,
    stride: int = 4,
    train_fraction: float = 0.80,
    guard_gap: int = 15,
    clients: tuple[str, ...] = HAI_CLIENT_PROCESSES,
    anchor_process: str = HAI_ANCHOR_PROCESS,
    max_train_rows: int | None = 200_000,
) -> FederationDataset:
    raw = read_hai_raw(data_dir, max_train_rows=max_train_rows)
    anchor_names = _process_columns(raw.feature_names, anchor_process)
    if not anchor_names:
        raise ValueError(f"anchor process {anchor_process!r} has no feature columns")

    split = attach_test_range(
        split_normal_train_cal(
            len(raw.normal_values),
            train_fraction=train_fraction,
            guard_gap=guard_gap,
        ),
        n_test_rows=len(raw.test_values),
    )

    feature_clients: list[ClientFeatures] = []

    for client_id in clients:
        aux_names = _process_columns(raw.feature_names, client_id)
        if not aux_names:
            raise ValueError(f"client process {client_id!r} has no feature columns")

        feature_names = anchor_names + aux_names
        normal_block = _select_feature_block(raw.normal_values, raw.feature_names, feature_names)
        test_block = _select_feature_block(raw.test_values, raw.feature_names, feature_names)

        train_seg = windowize_segment(
            normal_block[split.train.start:split.train.end],
            raw.normal_timestamps[split.train.start:split.train.end],
            window_length,
            stride,
            raw_offset=split.train.start,
        )
        cal_seg = windowize_segment(
            normal_block[split.calibration.start:split.calibration.end],
            raw.normal_timestamps[split.calibration.start:split.calibration.end],
            window_length,
            stride,
            raw_offset=split.calibration.start,
        )
        if raw.test_value_parts:
            test_parts = []
            offset = 0
            for values_part, ts_part in zip(raw.test_value_parts, raw.test_timestamp_parts):
                block_part = _select_feature_block(values_part, raw.feature_names, feature_names)
                test_parts.append(windowize_segment(
                    block_part,
                    ts_part,
                    window_length,
                    stride,
                    raw_offset=offset,
                ))
                offset += len(values_part)
            test_values_win, test_indices = _concat_windowed(test_parts)
            test_seg = WindowedSegment(values=test_values_win, indices=test_indices)
        else:
            test_seg = windowize_segment(
                test_block,
                raw.test_timestamps,
                window_length,
                stride,
                raw_offset=0,
            )
        scaler = TrainOnlyStandardizer.fit(train_seg.values)

        node_specs = tuple(
            NodeSpec(
                local_node_id=name,
                semantic_id=name,
                source_series_id=f"HAI21.03:{name}",
                role="anchor",
            )
            for name in anchor_names
        ) + tuple(
            NodeSpec(
                local_node_id=name,
                semantic_id=name,
                source_series_id=f"HAI21.03:{name}",
                role="auxiliary",
            )
            for name in aux_names
        )
        client_spec = ClientSpec(
            client_id=client_id,
            site_id="HAI21.03-single-testbed",
            node_specs=node_specs,
        )
        feature_clients.append(ClientFeatures(
            client_id=client_id,
            feature_names=feature_names,
            anchor_names=anchor_names,
            aux_names=aux_names,
            train_x=scaler.transform(train_seg.values),
            calibration_x=scaler.transform(cal_seg.values),
            test_x=scaler.transform(test_seg.values),
            train_index=train_seg.indices,
            calibration_index=cal_seg.indices,
            test_index=test_seg.indices,
            client_spec=client_spec,
            graph_metadata={
                "federation_type": "shared_context_vertical",
                "shared_anchor_observations": True,
                "normalization": scaler.fit_scope,
            },
        ))
    federation = FederationDataset(
        dataset="hai",
        protocol_version=2,
        federation_type="shared_context_vertical",
        shared_anchor_observations=True,
        clients=tuple(feature_clients),
        n_anchor=len(anchor_names),
        anchor_names=anchor_names,
        temporal_split=split,
        window_length=window_length,
        stride=stride,
    )
    validate_protocol(federation)
    return federation


def build_wadi_shared_context_protocol(
    data_dir: str | Path,
    *,
    window_length: int = 16,
    stride: int = 4,
    train_fraction: float = 0.80,
    guard_gap: int = 15,
    max_train_rows: int | None = 120_000,
) -> FederationDataset:
    data_dir = Path(data_dir)
    normal_df = _clean_columns(pd.read_csv(data_dir / "WADI_14days_new.csv", low_memory=False, nrows=max_train_rows))
    test_df = _clean_columns(pd.read_csv(data_dir / "WADI_attackdataLABLE.csv", header=1, low_memory=False))
    non_feature = {"row", "date", "time", "attack", "lable", "label", "index"}
    feature_names = tuple(
        c for c in normal_df.columns
        if c in test_df.columns and not any(k in c.lower() for k in non_feature)
    )
    anchor_names = tuple(c for c in WADI_ANCHORS if c in feature_names)
    if not anchor_names:
        anchor_names = tuple(c for c in feature_names if c.startswith("3_"))[:8]
    split = attach_test_range(
        split_normal_train_cal(len(normal_df), train_fraction=train_fraction, guard_gap=guard_gap),
        n_test_rows=len(test_df),
    )
    normal_values = _numeric_frame(normal_df, feature_names)
    test_values = _numeric_frame(test_df, feature_names)
    timestamps_train = np.arange(len(normal_df), dtype=np.int64)
    timestamps_test = np.arange(len(test_df), dtype=np.int64)
    specs = {
        "Zone1": tuple(c for c in feature_names if c.startswith("1_")),
        "Zone2": tuple(c for c in feature_names if c.startswith(("2_", "2A_", "2B_"))),
    }
    clients: list[ClientFeatures] = []
    for client_id, aux_names in specs.items():
        selected = anchor_names + tuple(c for c in aux_names if c not in anchor_names)
        block = _select_feature_block(normal_values, feature_names, selected)
        test_block = _select_feature_block(test_values, feature_names, selected)
        clients.append(_build_client(
            dataset="wadi",
            site_id="WADI-single-testbed",
            client_id=client_id,
            feature_names=selected,
            anchor_names=anchor_names,
            aux_names=tuple(c for c in selected if c not in anchor_names),
            train_values=block[split.train.start:split.train.end],
            train_timestamps=timestamps_train[split.train.start:split.train.end],
            train_offset=split.train.start,
            cal_values=block[split.calibration.start:split.calibration.end],
            cal_timestamps=timestamps_train[split.calibration.start:split.calibration.end],
            cal_offset=split.calibration.start,
            test_values=test_block,
            test_timestamps=timestamps_test,
            window_length=window_length,
            stride=stride,
        ))
    return _federation(dataset="wadi", clients=clients, n_anchor=len(anchor_names), anchor_names=anchor_names, split=split, window_length=window_length, stride=stride)


def build_swat_shared_context_protocol(
    data_dir: str | Path,
    *,
    window_length: int = 16,
    stride: int = 4,
    train_fraction: float = 0.80,
    guard_gap: int = 15,
    max_train_rows: int | None = None,
) -> FederationDataset:
    data_dir = Path(data_dir)
    normal_df = _clean_columns(pd.read_csv(data_dir / "normal.csv", low_memory=False, nrows=max_train_rows))
    test_df = _clean_columns(pd.read_csv(data_dir / "merged.csv", low_memory=False))
    feature_names = tuple(c for c in normal_df.columns if c not in {"Timestamp", "Normal/Attack"} and c in test_df.columns)
    anchor_names = tuple(c for c in SWAT_ANCHORS if c in feature_names)
    split = attach_test_range(
        split_normal_train_cal(len(normal_df), train_fraction=train_fraction, guard_gap=guard_gap),
        n_test_rows=len(test_df),
    )
    normal_values = _numeric_frame(normal_df, feature_names)
    test_values = _numeric_frame(test_df, feature_names)
    timestamps_train = np.arange(len(normal_df), dtype=np.int64)
    timestamps_test = np.arange(len(test_df), dtype=np.int64)
    clients: list[ClientFeatures] = []
    for client_id, local_cols in SWAT_CLIENTS.items():
        aux_names = tuple(c for c in local_cols if c in feature_names and c not in anchor_names)
        selected = anchor_names + aux_names
        block = _select_feature_block(normal_values, feature_names, selected)
        test_block = _select_feature_block(test_values, feature_names, selected)
        clients.append(_build_client(
            dataset="swat",
            site_id="SWaT-single-testbed",
            client_id=client_id,
            feature_names=selected,
            anchor_names=anchor_names,
            aux_names=aux_names,
            train_values=block[split.train.start:split.train.end],
            train_timestamps=timestamps_train[split.train.start:split.train.end],
            train_offset=split.train.start,
            cal_values=block[split.calibration.start:split.calibration.end],
            cal_timestamps=timestamps_train[split.calibration.start:split.calibration.end],
            cal_offset=split.calibration.start,
            test_values=test_block,
            test_timestamps=timestamps_test,
            window_length=window_length,
            stride=stride,
        ))
    return _federation(dataset="swat", clients=clients, n_anchor=len(anchor_names), anchor_names=anchor_names, split=split, window_length=window_length, stride=stride)


def build_batadal_shared_context_protocol(
    data_dir: str | Path,
    *,
    window_length: int = 32,
    stride: int = 1,
    train_fraction: float = 0.80,
    guard_gap: int = 0,
    max_train_rows: int | None = None,
) -> FederationDataset:
    data_dir = Path(data_dir)
    normal_df = _clean_columns(pd.read_csv(data_dir / "BATADAL_dataset03.csv", skipinitialspace=True, nrows=max_train_rows))
    test_df = _clean_columns(pd.read_csv(data_dir / "BATADAL_dataset04.csv", skipinitialspace=True))
    feature_names = tuple(c for c in normal_df.columns if c not in {"DATETIME", "ATT_FLAG"} and c in test_df.columns)
    anchor_names = tuple(c for c in BATADAL_ANCHORS if c in feature_names)
    split = attach_test_range(
        split_normal_train_cal(len(normal_df), train_fraction=train_fraction, guard_gap=guard_gap),
        n_test_rows=len(test_df),
    )
    normal_values = _numeric_frame(normal_df, feature_names)
    test_values = _numeric_frame(test_df, feature_names)
    timestamps_train = np.arange(len(normal_df), dtype=np.int64)
    timestamps_test = np.arange(len(test_df), dtype=np.int64)
    clients: list[ClientFeatures] = []
    for client_id, aux_source in BATADAL_CLIENTS.items():
        aux_names = tuple(c for c in aux_source if c in feature_names)
        selected = anchor_names + aux_names
        block = _select_feature_block(normal_values, feature_names, selected)
        test_block = _select_feature_block(test_values, feature_names, selected)
        clients.append(_build_client(
            dataset="batadal",
            site_id="BATADAL-single-testbed",
            client_id=client_id,
            feature_names=selected,
            anchor_names=anchor_names,
            aux_names=aux_names,
            train_values=block[split.train.start:split.train.end],
            train_timestamps=timestamps_train[split.train.start:split.train.end],
            train_offset=split.train.start,
            cal_values=block[split.calibration.start:split.calibration.end],
            cal_timestamps=timestamps_train[split.calibration.start:split.calibration.end],
            cal_offset=split.calibration.start,
            test_values=test_block,
            test_timestamps=timestamps_test,
            window_length=window_length,
            stride=stride,
        ))
    return _federation(dataset="batadal", clients=clients, n_anchor=len(anchor_names), anchor_names=anchor_names, split=split, window_length=window_length, stride=stride)


def build_protocol(dataset: str, data_dir: str | Path, **kwargs) -> FederationDataset:
    if dataset == "hai":
        return build_hai_shared_context_protocol(data_dir, **kwargs)
    if dataset == "wadi":
        return build_wadi_shared_context_protocol(data_dir, **kwargs)
    if dataset == "swat":
        return build_swat_shared_context_protocol(data_dir, **kwargs)
    if dataset == "batadal":
        return build_batadal_shared_context_protocol(data_dir, **kwargs)
    raise ValueError(f"unknown dataset: {dataset!r}")


def to_model_clients(federation: FederationDataset) -> tuple[list[dict], int, list[str]]:
    model_clients: list[dict] = []
    for client in federation.clients:
        item = {
            "client_name": client.client_id,
            "feature_names": list(client.feature_names),
            "anchor_names": list(client.anchor_names),
            "aux_names": list(client.aux_names),
            "X_train": client.train_x,
            "X_cal": client.calibration_x,
            "X_test": client.test_x,
            "n_k": client.train_x.shape[2],
            "protocol_version": federation.protocol_version,
            "federation_type": federation.federation_type,
            "window_indices": {
                "train": client.train_index,
                "calibration": client.calibration_index,
                "test": client.test_index,
            },
            "client_spec": client.client_spec,
        }
        model_clients.append(item)
    return model_clients, federation.n_anchor, list(federation.anchor_names)
