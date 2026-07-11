"""Window construction utilities."""

from __future__ import annotations

import numpy as np

from .schema import WindowIndex, WindowedSegment


def count_windows(n_rows: int, window_length: int, stride: int) -> int:
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if n_rows < window_length:
        return 0
    return (n_rows - window_length) // stride + 1


def windowize_segment(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_length: int,
    stride: int,
    raw_offset: int = 0,
) -> WindowedSegment:
    if values.shape[0] != len(timestamps):
        raise ValueError("values and timestamps must have the same row count")
    windows: list[np.ndarray] = []
    indices: list[WindowIndex] = []
    for start in range(0, len(values) - window_length + 1, stride):
        end = start + window_length
        windows.append(values[start:end])
        indices.append(WindowIndex(
            raw_start=raw_offset + start,
            raw_end=raw_offset + end,
            start_timestamp=timestamps[start],
            end_timestamp=timestamps[end - 1],
        ))
    if windows:
        x = np.asarray(windows, dtype=np.float32)
    else:
        x = np.empty((0, window_length, values.shape[1], values.shape[2]), dtype=np.float32)
    return WindowedSegment(values=x, indices=tuple(indices))

