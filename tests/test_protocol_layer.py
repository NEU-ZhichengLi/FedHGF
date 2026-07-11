from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fedgad_full import FedGAD, estimate_anchor_aux_correlation, estimate_aux_correlation
from src.fedhgf.calibration import QuantileCalibrator
from src.fedhgf.data.normalization import TrainOnlyStandardizer
from src.fedhgf.data.protocol_builder import build_hai_shared_context_protocol, to_model_clients
from src.fedhgf.data.schema import (
    ClientFeatures,
    ClientSpec,
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
    windowize_segment,
)
from src.fedhgf.evaluation import load_hai_test_labels, windowize_labels_for_evaluation
from src.fedhgf.federation import AssumedSecAggAggregator, ClientMessage, DPSimulatedAggregator, PlainAggregator


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
    def build_client_ids() -> tuple[str, ...]:
        return ("P1", "P2", "P4")

    assert build_client_ids() == ("P1", "P2", "P4")


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
        n_anchor=1,
        anchor_names=("a",),
        temporal_split=TemporalSplit(TemporalRange(0, 1), TemporalRange(1, 2), TemporalRange(0, 1)),
        window_length=2,
        stride=1,
    )
    validate_anchor_semantics(fed)

    bad = FederationDataset(
        dataset=fed.dataset,
        protocol_version=fed.protocol_version,
        federation_type=fed.federation_type,
        shared_anchor_observations=False,
        clients=(_synthetic_client("c1", "same:a"), _synthetic_client("c2", "same:a")),
        n_anchor=fed.n_anchor,
        anchor_names=fed.anchor_names,
        temporal_split=fed.temporal_split,
        window_length=fed.window_length,
        stride=fed.stride,
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
        n_anchor=1,
        anchor_names=("a",),
        temporal_split=TemporalSplit(TemporalRange(0, 1), TemporalRange(1, 2), TemporalRange(0, 1)),
        window_length=2,
        stride=1,
    )
    validate_model_features_are_label_free(fed)


def _write_hai_fixture(root: Path, test_attack: np.ndarray) -> None:
    n_train = 24
    n_test = len(test_attack)
    train = pd.DataFrame({
        "time": np.arange(n_train),
        "attack": np.zeros(n_train, dtype=int),
        "attack_P1": np.zeros(n_train, dtype=int),
        "attack_P2": np.zeros(n_train, dtype=int),
        "attack_P3": np.zeros(n_train, dtype=int),
        "P3_A": np.arange(n_train, dtype=float),
        "P1_A": np.arange(n_train, dtype=float) + 10,
        "P2_A": np.arange(n_train, dtype=float) + 20,
        "P4_A": np.arange(n_train, dtype=float) + 40,
    })
    test = pd.DataFrame({
        "time": np.arange(n_test),
        "attack": test_attack.astype(int),
        "attack_P1": test_attack.astype(int),
        "attack_P2": test_attack.astype(int),
        "attack_P3": test_attack.astype(int),
        "P3_A": np.arange(n_test, dtype=float),
        "P1_A": np.arange(n_test, dtype=float) + 10,
        "P2_A": np.arange(n_test, dtype=float) + 20,
        "P4_A": np.arange(n_test, dtype=float) + 40,
    })
    train.to_csv(root / "train1.csv.gz", index=False, compression="gzip")
    test.to_csv(root / "test1.csv.gz", index=False, compression="gzip")


def _model_snapshot(model_clients: list[dict]) -> tuple:
    return tuple(
        (
            client["client_name"],
            tuple(client["feature_names"]),
            client["X_train"].shape,
            client["X_cal"].shape,
            client["X_test"].shape,
            float(client["X_train"].sum()),
            float(client["X_cal"].sum()),
            float(client["X_test"].sum()),
        )
        for client in model_clients
    )


def test_no_label_used_in_protocol_builder():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        labels_a = np.array([0, 1, 0, 0, 1, 0, 0, 1], dtype=np.int64)
        labels_b = np.zeros_like(labels_a)
        _write_hai_fixture(Path(a), labels_a)
        _write_hai_fixture(Path(b), labels_b)
        fed_a = build_hai_shared_context_protocol(a, window_length=4, stride=2, guard_gap=3, max_train_rows=None)
        fed_b = build_hai_shared_context_protocol(b, window_length=4, stride=2, guard_gap=3, max_train_rows=None)
        clients_a, _, _ = to_model_clients(fed_a)
        clients_b, _, _ = to_model_clients(fed_b)
        assert _model_snapshot(clients_a) == _model_snapshot(clients_b)
        assert not hasattr(fed_a, "labels")
        assert windowize_labels_for_evaluation(load_hai_test_labels(a), 4, 2).sum() > 0
        assert windowize_labels_for_evaluation(load_hai_test_labels(b), 4, 2).sum() == 0


def test_protocol_preserves_complete_test_axis_and_natural_rate():
    with tempfile.TemporaryDirectory() as root:
        labels = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=np.int64)
        _write_hai_fixture(Path(root), labels)
        fed = build_hai_shared_context_protocol(root, window_length=4, stride=2, guard_gap=3, max_train_rows=None)
        expected_starts = tuple(range(0, len(labels) - 4 + 1, 2))
        for client in fed.clients:
            assert tuple(w.raw_start for w in client.test_index) == expected_starts
            assert tuple(w.raw_end for w in client.test_index) == tuple(s + 4 for s in expected_starts)
            assert len(client.test_x) == count_windows(len(labels), 4, 2)

        y = windowize_labels_for_evaluation(load_hai_test_labels(root), 4, 2)
        assert len(y) == count_windows(len(labels), 4, 2)
        assert float(y.mean()) == float(np.mean([1, 1, 1]))


def test_fedgad_score_contract_has_no_labels_or_thresholds():
    model = FedGAD(n_anchor=1, device="cpu")
    model._branch_scores = lambda k, c, split, center_np=None: (
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
    )
    model.client_cal = {0: {
        "s1": np.array([0.0, 1.0], dtype=np.float32),
        "s2": np.array([1.0, 2.0], dtype=np.float32),
        "s3": np.array([2.0, 3.0], dtype=np.float32),
    }}
    model.client_fusion_weights = {0: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)}
    out = model.score([{"client_name": "P1"}], split="test")[0]
    assert "score" in out and "evidence" in out
    assert "y_true" not in out and "y_pred" not in out and "tau" not in out


def test_dp_aggregator_is_simulated_and_records_server_visible_event():
    messages = [
        ClientMessage("c1", 1, "encoder", torch.ones(2), 2),
        ClientMessage("c2", 1, "encoder", torch.zeros(2), 2),
    ]
    plain = PlainAggregator().aggregate(messages)
    assert torch.allclose(plain, torch.tensor([0.5, 0.5]))

    dp = DPSimulatedAggregator(clip_norm=1.0, noise_multiplier=0.0)
    out = dp.aggregate(messages)
    assert len(dp.accountant.events) == 1
    assert dp.accountant.events[0].channel == "encoder"
    assert dp.backend == "dp_simulator"
    assert out.shape == plain.shape

    assumed = AssumedSecAggAggregator()
    assert assumed.backend == "assumed"
    assert torch.allclose(assumed.aggregate(messages), plain)


def test_local_auxiliary_graphs_do_not_accept_dp_noise():
    for fn in (estimate_aux_correlation, estimate_anchor_aux_correlation):
        sig = inspect.signature(fn)
        assert "use_dp" not in sig.parameters
        assert "sigma_g" not in sig.parameters
        assert "C_g" not in sig.parameters

    rng = np.random.RandomState(7)
    x = rng.normal(size=(8, 5, 4, 1)).astype(np.float32)
    aux_a = estimate_aux_correlation(x, n_anchor=1)
    aux_b = estimate_aux_correlation(x, n_anchor=1)
    anchor_aux_a = estimate_anchor_aux_correlation(x, n_anchor=1)
    anchor_aux_b = estimate_anchor_aux_correlation(x, n_anchor=1)
    assert np.allclose(aux_a, aux_b)
    assert np.allclose(anchor_aux_a, anchor_aux_b)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("protocol layer tests passed")
