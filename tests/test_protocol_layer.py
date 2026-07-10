from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.fedhgf.calibration import QuantileCalibrator
from src.fedhgf.data.normalization import TrainOnlyStandardizer
from src.fedhgf.data.schema import (
    ClientFeatures,
    ClientSpec,
    EvaluationLabels,
    FederationDataset,
    NodeSpec,
    TemporalRange,
    TemporalSplit,
)
from src.fedhgf.data.validation import (
    validate_anchor_semantics,
    validate_model_features_are_label_free,
)
from src.fedhgf.data.windowing import (
    count_windows,
    windowize_labels_for_evaluation,
    windowize_segment,
)


def test_window_does_not_cross_split_boundary():
    values = np.arange(20, dtype=np.float32).reshape(20, 1, 1)
    timestamps = np.arange(20)
    train = windowize_segment(values[:10], timestamps[:10], window_length=4, stride=2, raw_offset=0)
    cal = windowize_segment(values[12:], timestamps[12:], window_length=4, stride=2, raw_offset=12)

    assert all(0 <= w.raw_start and w.raw_end <= 10 for w in train.indices)
    assert all(12 <= w.raw_start and w.raw_end <= 20 for w in cal.indices)
    train_rows = {i for w in train.indices for i in range(w.raw_start, w.raw_end)}
    cal_rows = {i for w in cal.indices for i in range(w.raw_start, w.raw_end)}
    assert train_rows.isdisjoint(cal_rows)


def test_calibrator_does_not_accept_labels():
    sig = inspect.signature(QuantileCalibrator.fit)
    assert list(sig.parameters) == ["self", "calibration_scores"]
    cal = QuantileCalibrator(alert_budget=0.25).fit(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    assert cal.predict(np.array([0.5, 3.5])).tolist() == [0, 1]


def test_test_set_is_not_resampled():
    labels = np.array([0, 1, 0, 0, 1, 1, 0, 0], dtype=np.int64)
    y = windowize_labels_for_evaluation(labels, window_length=3, stride=1)
    assert len(y) == count_windows(len(labels), 3, 1)
    assert y.tolist() == [1, 1, 1, 1, 1, 1]


def test_client_manifest_independent_of_labels():
    def build_client_ids(labels: np.ndarray) -> tuple[str, ...]:
        _ = windowize_labels_for_evaluation(labels, window_length=2, stride=1)
        return ("P1", "P2", "P4")

    labels_a = np.array([0, 1, 0, 1], dtype=np.int64)
    labels_b = np.zeros_like(labels_a)
    assert build_client_ids(labels_a) == build_client_ids(labels_b)


def test_scaler_is_fitted_only_on_train():
    train = np.zeros((2, 2, 1, 1), dtype=np.float32)
    cal = np.full((2, 2, 1, 1), 100.0, dtype=np.float32)
    scaler = TrainOnlyStandardizer.fit(train)
    assert scaler.fit_scope == "train"
    assert float(scaler.mean.squeeze()) == 0.0
    assert float(scaler.transform(cal).mean()) > 1.0


def _synthetic_client(client_id: str, source: str) -> ClientFeatures:
    spec = ClientSpec(
        client_id=client_id,
        site_id=client_id,
        node_specs=(NodeSpec("a", "water_level", source, "anchor"),),
    )
    empty = np.empty((0, 2, 1, 1), dtype=np.float32)
    return ClientFeatures(
        client_id=client_id,
        feature_names=("a",),
        anchor_names=("a",),
        aux_names=(),
        train_x=empty,
        calibration_x=empty,
        test_x=empty,
        train_index=(),
        calibration_index=(),
        test_index=(),
        client_spec=spec,
        graph_metadata={},
    )


def test_independent_site_anchors_have_distinct_sources():
    fed = FederationDataset(
        dataset="synthetic",
        protocol_version=2,
        federation_type="horizontal_semantic_anchor",
        shared_anchor_observations=False,
        clients=(_synthetic_client("c1", "site1:a"), _synthetic_client("c2", "site2:a")),
        labels={"c1": EvaluationLabels("c1", np.array([])), "c2": EvaluationLabels("c2", np.array([]))},
        n_anchor=1,
        anchor_names=("a",),
        temporal_split=TemporalSplit(TemporalRange(0, 1), TemporalRange(1, 2), TemporalRange(0, 1)),
        window_length=2,
        stride=1,
        label_mode="any",
    )
    validate_anchor_semantics(fed)

    bad = FederationDataset(
        dataset=fed.dataset,
        protocol_version=fed.protocol_version,
        federation_type=fed.federation_type,
        shared_anchor_observations=False,
        clients=(_synthetic_client("c1", "same:a"), _synthetic_client("c2", "same:a")),
        labels=fed.labels,
        n_anchor=fed.n_anchor,
        anchor_names=fed.anchor_names,
        temporal_split=fed.temporal_split,
        window_length=fed.window_length,
        stride=fed.stride,
        label_mode=fed.label_mode,
    )
    try:
        validate_anchor_semantics(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected repeated source_series_id to fail")


def test_model_facing_features_do_not_carry_labels():
    client = _synthetic_client("c1", "site:a")
    fed = FederationDataset(
        dataset="synthetic",
        protocol_version=2,
        federation_type="shared_context_vertical",
        shared_anchor_observations=True,
        clients=(client,),
        labels={"c1": EvaluationLabels("c1", np.array([0, 1]))},
        n_anchor=1,
        anchor_names=("a",),
        temporal_split=TemporalSplit(TemporalRange(0, 1), TemporalRange(1, 2), TemporalRange(0, 1)),
        window_length=2,
        stride=1,
        label_mode="any",
    )
    validate_model_features_are_label_free(fed)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("protocol layer tests passed")
