"""Chronological split helpers."""

from __future__ import annotations

from .schema import TemporalRange, TemporalSplit


def split_normal_train_cal(
    n_rows: int,
    train_fraction: float = 0.80,
    guard_gap: int = 0,
) -> TemporalSplit:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if guard_gap < 0:
        raise ValueError("guard_gap must be non-negative")
    train_end = int(round(n_rows * train_fraction))
    cal_start = min(n_rows, train_end + guard_gap)
    if train_end <= 0 or cal_start >= n_rows:
        raise ValueError("split leaves an empty train or calibration segment")
    return TemporalSplit(
        train=TemporalRange(0, train_end),
        calibration=TemporalRange(cal_start, n_rows),
        test=TemporalRange(0, 0),
    )


def attach_test_range(split: TemporalSplit, n_test_rows: int) -> TemporalSplit:
    return TemporalSplit(
        train=split.train,
        calibration=split.calibration,
        test=TemporalRange(0, n_test_rows),
    )

