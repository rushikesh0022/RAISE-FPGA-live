"""Leakage-safe time-grid alignment and history construction.

The functions in this module operate on one telemetry table containing explicit
run identifiers.  They intentionally know nothing about event thresholds or
batch-summary tables.  In particular, ``batch_id`` is metadata and is never an
input feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


_COMPLETED_BATCH_PREFIXES = ("avg_", "mean_", "peak_", "energy_")
_COMPLETED_BATCH_FIELDS = {
    "batch_accuracy",
    "fn",
    "fp",
    "hw_duration_s",
    "hw_n_samples",
    "inference_time_ms",
    "n_samples",
    "start_sample_idx",
    "tn",
    "tp",
}


def _reject_noncausal_features(features: Sequence[str]) -> None:
    forbidden = [
        name
        for name in features
        if name == "batch_id"
        or name in _COMPLETED_BATCH_FIELDS
        or name.startswith(_COMPLETED_BATCH_PREFIXES)
    ]
    if forbidden:
        raise ValueError(
            "completed-batch/association fields cannot be input features: "
            f"{sorted(forbidden)}"
        )


@dataclass(frozen=True)
class WindowSpec:
    target_hz: float = 10.0
    history_samples: int = 33
    max_staleness_s: float = 0.2
    max_gap_s: float = 0.5

    def __post_init__(self) -> None:
        if self.target_hz <= 0 or self.history_samples < 2:
            raise ValueError("target_hz must be positive and history_samples >= 2")
        if self.max_staleness_s <= 0 or self.max_gap_s <= 0:
            raise ValueError("staleness and gap limits must be positive")

    @property
    def step_s(self) -> float:
        return 1.0 / self.target_hz

    @property
    def history_span_s(self) -> float:
        return (self.history_samples - 1) * self.step_s


@dataclass
class WindowBatch:
    """Model histories plus traceable timing and identity metadata."""

    values: np.ndarray  # [N, history_samples, features]
    metadata: pd.DataFrame
    feature_columns: tuple[str, ...]
    spec: WindowSpec


@dataclass(frozen=True)
class StandardizationStats:
    feature_columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.feature_columns):
            raise ValueError("feature dimension does not match scaler")
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)


def fit_training_standardizer(
    values: np.ndarray,
    *,
    feature_columns: Sequence[str],
    partitions: Sequence[str],
) -> StandardizationStats:
    """Fit standardization only when every supplied window is training data."""

    parts = np.asarray(partitions, dtype=object)
    if values.shape[0] != len(parts):
        raise ValueError("one partition is required per window")
    non_training = sorted(set(parts.tolist()) - {"training"})
    if non_training:
        raise ValueError(f"refusing scaler fit on non-training partitions: {non_training}")
    flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    mean = np.nanmean(flat, axis=0)
    scale = np.nanstd(flat, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return StandardizationStats(tuple(feature_columns), mean, scale)


def deduplicate_telemetry(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    run_col: str = "run_id",
    timestamp_col: str = "timestamp_s",
    identity_columns: Sequence[str] = ("workload_id", "rep_id"),
) -> pd.DataFrame:
    """Keep one consistent sensor observation per run/timestamp.

    Duplicate rows may have different ``batch_id`` values.  Feature and stable
    identity disagreement is rejected instead of being resolved silently.
    """

    features = tuple(feature_columns)
    _reject_noncausal_features(features)
    required = {run_col, timestamp_col, *features, *identity_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    if frame[timestamp_col].isna().any():
        raise ValueError("timestamps must not be missing")

    stable = [*identity_columns, *features]
    duplicate_mask = frame.duplicated([run_col, timestamp_col], keep=False)
    if duplicate_mask.any():
        duplicate_rows = frame.loc[
            duplicate_mask, [run_col, timestamp_col, *stable]
        ]
        diversity = duplicate_rows.groupby(
            [run_col, timestamp_col], sort=False, dropna=False
        )[stable].nunique(dropna=False)
        bad = diversity.gt(1)
        if bad.to_numpy().any():
            examples = []
            for key, column in zip(*np.nonzero(bad.to_numpy()), strict=True):
                group_key = bad.index[key]
                examples.append((*group_key, bad.columns[column]))
                if len(examples) == 5:
                    break
            raise ValueError(f"inconsistent duplicate telemetry: {examples}")

    keep = [run_col, *identity_columns, timestamp_col, *features]
    return (
        frame[keep]
        .drop_duplicates([run_col, timestamp_col], keep="last")
        .sort_values([run_col, timestamp_col], kind="stable")
        .reset_index(drop=True)
    )


def _grid_bounds(first: float, last: float, step: float) -> np.ndarray:
    # Integer ticks avoid accumulated floating-point drift.
    lo = int(np.ceil((first - 1e-12) / step))
    hi = int(np.floor((last + 1e-12) / step))
    if hi < lo:
        return np.empty(0, dtype=np.float64)
    return np.arange(lo, hi + 1, dtype=np.int64) * step


def build_windows(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    spec: WindowSpec = WindowSpec(),
    run_col: str = "run_id",
    timestamp_col: str = "timestamp_s",
    identity_columns: Sequence[str] = ("workload_id", "rep_id"),
    include_invalid: bool = False,
) -> WindowBatch:
    """Causally align telemetry and construct 33-sample histories.

    Every grid value is selected by backward as-of lookup and must be no more
    than ``max_staleness_s`` old.  A history is invalid if it crosses a raw
    source gap larger than ``max_gap_s``.  Processing is isolated by run, so a
    history can never cross run or repetition boundaries.
    """

    clean = deduplicate_telemetry(
        frame,
        feature_columns=feature_columns,
        run_col=run_col,
        timestamp_col=timestamp_col,
        identity_columns=identity_columns,
    )
    features = tuple(feature_columns)
    all_values: list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []

    for run_id, group in clean.groupby(run_col, sort=False, dropna=False):
        source_t = group[timestamp_col].to_numpy(dtype=np.float64)
        if len(source_t) < 2:
            continue
        source_x = group.loc[:, features].to_numpy(dtype=np.float64)
        grid = _grid_bounds(source_t[0], source_t[-1], spec.step_s)
        if len(grid) < spec.history_samples:
            continue

        source_index = np.searchsorted(source_t, grid, side="right") - 1
        has_source = source_index >= 0
        safe_index = np.maximum(source_index, 0)
        staleness = grid - source_t[safe_index]
        aligned_valid = (
            has_source
            & (staleness >= -1e-9)
            & (staleness <= spec.max_staleness_s + 1e-12)
        )
        aligned = source_x[safe_index]
        aligned[~aligned_valid] = np.nan

        # Segment numbers increment after a forbidden raw gap.  All source
        # observations used in a valid history must belong to one segment.
        segments = np.zeros(len(source_t), dtype=np.int64)
        segments[1:] = np.cumsum(np.diff(source_t) > spec.max_gap_s + 1e-12)
        aligned_segment = segments[safe_index]

        windows = np.lib.stride_tricks.sliding_window_view(
            aligned, spec.history_samples, axis=0
        ).transpose(0, 2, 1)
        source_windows = np.lib.stride_tricks.sliding_window_view(
            safe_index, spec.history_samples
        )
        valid_windows = np.lib.stride_tricks.sliding_window_view(
            aligned_valid, spec.history_samples
        )
        segment_windows = np.lib.stride_tricks.sliding_window_view(
            aligned_segment, spec.history_samples
        )
        valid = valid_windows.all(axis=1)
        valid &= segment_windows[:, 0] == segment_windows[:, -1]
        valid &= np.isfinite(windows).all(axis=(1, 2))

        anchors = grid[spec.history_samples - 1 :]
        last_source_index = source_windows[:, -1]
        segment_end_indices = np.r_[
            np.flatnonzero(np.diff(segments) != 0), len(source_t) - 1
        ]
        segment_end_by_source = segment_end_indices[segments]
        followup_end = source_t[segment_end_by_source[last_source_index]]

        reason = np.full(len(valid), "", dtype=object)
        reason[~valid_windows.all(axis=1)] = "stale_or_missing_history_sample"
        crossed = segment_windows[:, 0] != segment_windows[:, -1]
        reason[crossed] = "history_crosses_source_gap"
        nonfinite = ~np.isfinite(windows).all(axis=(1, 2))
        reason[nonfinite & (reason == "")] = "nonfinite_feature"

        identities = {column: group[column].iloc[0] for column in identity_columns}
        meta = pd.DataFrame(
            {
                run_col: run_id,
                **identities,
                "anchor_timestamp": anchors,
                "latest_source_timestamp": source_t[last_source_index],
                "latest_staleness_s": anchors - source_t[last_source_index],
                "followup_end_timestamp": followup_end,
                "input_valid": valid,
                "invalid_reason": reason,
            }
        )
        if not include_invalid:
            windows = windows[valid]
            meta = meta.loc[valid].reset_index(drop=True)
        all_values.append(windows.astype(np.float32, copy=True))
        all_meta.append(meta)

    shape = (0, spec.history_samples, len(features))
    values = np.concatenate(all_values, axis=0) if all_values else np.empty(shape, np.float32)
    metadata = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    return WindowBatch(values, metadata, features, spec)
