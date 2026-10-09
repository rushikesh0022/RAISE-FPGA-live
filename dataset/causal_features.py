"""Leakage-safe engineered predictors and dense future-temperature targets.

The neural model's 33-sample core remains the source of short-context features.
This module adds only compact summaries; it never appends future targets to the
predictor matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from dataset.build_windows import WindowBatch, deduplicate_telemetry


@dataclass
class CausalFeatureBatch:
    values: np.ndarray
    columns: tuple[str, ...]
    metadata: pd.DataFrame


def _time_projection(width: int, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights mapping evenly spaced values to slope and second derivative."""

    t = np.arange(width, dtype=np.float64) * step_s
    linear = np.linalg.pinv(np.column_stack((np.ones(width), t)))[1]
    quadratic = np.linalg.pinv(np.column_stack((np.ones(width), t, t * t)))[2] * 2.0
    return linear, quadratic


def core_temperature_dynamics(
    windows: WindowBatch,
    *,
    temperature_columns: Sequence[str],
    scale_samples: Sequence[int] = (5, 17, 33),
    run_start_s: Mapping[object, float] | None = None,
) -> CausalFeatureBatch:
    """Derive multi-scale slope, curvature and variability from core histories.

    Slopes use ordinary least squares on elapsed time, curvature is the fitted
    quadratic's second derivative, and variability is population standard
    deviation.  ``elapsed_since_run_start_s`` is included only when callers
    provide true raw run-start timestamps; it is never guessed from the first
    eligible window.
    """

    missing = set(temperature_columns).difference(windows.feature_columns)
    if missing:
        raise KeyError(f"temperature columns absent from core windows: {sorted(missing)}")
    widths = tuple(int(width) for width in scale_samples)
    if not widths or any(width < 3 or width > windows.spec.history_samples for width in widths):
        raise ValueError("scale widths must be between 3 and the core history length")
    if tuple(sorted(set(widths))) != widths:
        raise ValueError("scale widths must be unique and increasing")

    output: list[np.ndarray] = []
    names: list[str] = []
    if run_start_s is not None:
        starts = windows.metadata["run_id"].map(run_start_s)
        if starts.isna().any():
            missing_runs = windows.metadata.loc[starts.isna(), "run_id"].unique().tolist()
            raise KeyError(f"run start missing for: {missing_runs}")
        elapsed = windows.metadata["anchor_timestamp"].to_numpy(float) - starts.to_numpy(float)
        if np.any(elapsed < -1e-9):
            raise ValueError("anchor precedes declared run start")
        output.append(elapsed[:, None])
        names.append("elapsed_since_run_start_s")

    for column in temperature_columns:
        feature_index = windows.feature_columns.index(column)
        for width in widths:
            values = windows.values[:, -width:, feature_index].astype(np.float64)
            slope_w, curvature_w = _time_projection(width, windows.spec.step_s)
            output.extend(
                (
                    values @ slope_w[:, None],
                    values @ curvature_w[:, None],
                    np.std(values, axis=1, ddof=0)[:, None],
                )
            )
            span = (width - 1) * windows.spec.step_s
            suffix = f"{span:g}s"
            names.extend(
                (
                    f"{column}__slope_C_per_s__{suffix}",
                    f"{column}__curvature_C_per_s2__{suffix}",
                    f"{column}__std_C__{suffix}",
                )
            )

    matrix = np.concatenate(output, axis=1) if output else np.empty((len(windows.values), 0))
    return CausalFeatureBatch(
        matrix.astype(np.float32, copy=False), tuple(names), windows.metadata.copy()
    )


def long_context_temperature_summaries(
    telemetry: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    temperature_columns: Sequence[str],
    context_s: Sequence[float] = (10.0, 30.0),
    max_gap_s: float = 0.5,
) -> CausalFeatureBatch:
    """Compute optional causal summaries beyond the 3.2-second core.

    Each mean/std/slope uses raw samples in ``[anchor-context, anchor]`` from
    the anchor's current continuous segment only.  No value after the anchor is
    read.  Context with fewer than three samples is marked invalid.
    """

    contexts = tuple(float(value) for value in context_s)
    if not contexts or any(value <= 0 for value in contexts):
        raise ValueError("context durations must be positive")
    if tuple(sorted(set(contexts))) != contexts:
        raise ValueError("context durations must be unique and increasing")
    required_anchor = {"run_id", "anchor_timestamp"}
    if missing := required_anchor.difference(anchors.columns):
        raise KeyError(f"anchor columns missing: {sorted(missing)}")
    anchor_work = anchors.reset_index(drop=True)

    identity = tuple(c for c in ("workload_id", "rep_id") if c in telemetry.columns)
    clean = deduplicate_telemetry(
        telemetry, feature_columns=temperature_columns, identity_columns=identity
    )
    names = tuple(
        f"{column}__{stat}__context_{duration:g}s"
        for column in temperature_columns
        for duration in contexts
        for stat in ("mean_C", "std_C", "slope_C_per_s")
    )
    result = np.full((len(anchor_work), len(names)), np.nan, dtype=np.float64)

    for run_id, anchor_group in anchor_work.groupby("run_id", sort=False, dropna=False):
        source = clean.loc[clean.run_id == run_id]
        if source.empty:
            continue
        t = source.timestamp_s.to_numpy(float)
        x = source.loc[:, temperature_columns].to_numpy(float)
        segments = np.zeros(len(t), dtype=np.int64)
        segments[1:] = np.cumsum(np.diff(t) > max_gap_s + 1e-12)
        segment_starts = np.r_[0, np.flatnonzero(np.diff(segments) != 0) + 1]
        start_by_source = segment_starts[segments]
        anchor_t = anchor_group.anchor_timestamp.to_numpy(float)
        right = np.searchsorted(t, anchor_t, side="right")
        has_source = right > 0
        safe_source = np.maximum(right - 1, 0)
        segment_left = start_by_source[safe_source]
        row_indices = anchor_group.index.to_numpy()

        # Prefix moments make every arbitrary-duration summary O(1) per anchor.
        # This matters for the full corpus, where raw slicing per anchor would
        # repeatedly scan the same 10/30-second observations.
        for feature_index in range(len(temperature_columns)):
            y = x[:, feature_index]
            finite = np.isfinite(y)

            def prefix(value: np.ndarray) -> np.ndarray:
                return np.r_[0.0, np.cumsum(np.where(finite, value, 0.0))]

            count_p = np.r_[0, np.cumsum(finite.astype(np.int64))]
            y_p = prefix(y)
            yy_p = prefix(y * y)
            t_p = prefix(t)
            tt_p = prefix(t * t)
            ty_p = prefix(t * y)
            for duration_index, duration in enumerate(contexts):
                left = np.searchsorted(t, anchor_t - duration, side="left")
                left = np.maximum(left, segment_left)
                count = count_p[right] - count_p[left]
                sy = y_p[right] - y_p[left]
                syy = yy_p[right] - yy_p[left]
                st = t_p[right] - t_p[left]
                stt = tt_p[right] - tt_p[left]
                sty = ty_p[right] - ty_p[left]
                usable = has_source & (count >= 3)
                mean = np.divide(sy, count, out=np.full_like(sy, np.nan), where=usable)
                variance = np.divide(syy, count, out=np.full_like(syy, np.nan), where=usable) - mean * mean
                std = np.sqrt(np.maximum(variance, 0.0))
                denominator = stt - np.divide(st * st, count, out=np.zeros_like(st), where=count > 0)
                numerator = sty - np.divide(st * sy, count, out=np.zeros_like(st), where=count > 0)
                slope = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan),
                    where=usable & (denominator > 0),
                )
                output_col = (feature_index * len(contexts) + duration_index) * 3
                result[row_indices, output_col] = mean
                result[row_indices, output_col + 1] = std
                result[row_indices, output_col + 2] = slope

    metadata = anchor_work.copy()
    metadata["long_context_valid"] = np.isfinite(result).all(axis=1)
    return CausalFeatureBatch(result.astype(np.float32), names, metadata)


def make_future_temperature_targets(
    telemetry: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    temperature_col: str = "temp_pl_temp_C",
    horizons_s: Sequence[float] = (0.5, 2.0, 5.0),
    max_staleness_s: float = 0.2,
    max_gap_s: float = 0.5,
) -> pd.DataFrame:
    """Create masked future PL-temperature regression targets.

    A target uses the latest raw observation at or before the exact horizon
    boundary.  Its staleness must be bounded, it must occur after the anchor,
    and it must be in the same continuous raw segment.  The actual observation
    timestamp/effective horizon are retained for auditability.
    """

    horizons = np.asarray(horizons_s, dtype=float)
    if np.any(~np.isfinite(horizons)) or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0):
        raise ValueError("horizons must be finite, positive and increasing")
    required_anchor = {"run_id", "anchor_timestamp"}
    if missing := required_anchor.difference(anchors.columns):
        raise KeyError(f"anchor columns missing: {sorted(missing)}")
    identity = tuple(c for c in ("workload_id", "rep_id") if c in telemetry.columns)
    clean = deduplicate_telemetry(
        telemetry, feature_columns=[temperature_col], identity_columns=identity
    )
    output = anchors.reset_index(drop=False).rename(columns={"index": "anchor_index"})[
        ["anchor_index", "run_id", "anchor_timestamp"]
    ].copy()
    empty_columns: dict[str, np.ndarray] = {}
    for horizon in horizons:
        tag = f"{horizon:g}s".replace(".", "_")
        empty_columns[f"target_temp_pl_C__{tag}"] = np.full(len(output), np.nan)
        empty_columns[f"target_observed__{tag}"] = np.zeros(len(output), dtype=bool)
        empty_columns[f"target_timestamp__{tag}"] = np.full(len(output), np.nan)
        empty_columns[f"effective_horizon_s__{tag}"] = np.full(len(output), np.nan)
    output = pd.concat((output, pd.DataFrame(empty_columns)), axis=1)

    for run_id, anchor_group in output.groupby("run_id", sort=False, dropna=False):
        source = clean.loc[clean.run_id == run_id]
        if source.empty:
            continue
        t = source.timestamp_s.to_numpy(float)
        y = source[temperature_col].to_numpy(float)
        segments = np.zeros(len(t), dtype=np.int64)
        segments[1:] = np.cumsum(np.diff(t) > max_gap_s + 1e-12)
        anchor_t = anchor_group.anchor_timestamp.to_numpy(float)
        anchor_i = np.searchsorted(t, anchor_t, side="right") - 1
        anchor_exists = anchor_i >= 0
        safe_anchor_i = np.maximum(anchor_i, 0)
        for horizon in horizons:
            tag = f"{horizon:g}s".replace(".", "_")
            boundary = anchor_t + horizon
            target_i = np.searchsorted(t, boundary, side="right") - 1
            target_exists = target_i >= 0
            safe_target_i = np.maximum(target_i, 0)
            observed_t = t[safe_target_i]
            valid = (
                anchor_exists
                & target_exists
                & (observed_t > anchor_t)
                & (boundary - observed_t >= -1e-9)
                & (boundary - observed_t <= max_staleness_s + 1e-12)
                & (segments[safe_anchor_i] == segments[safe_target_i])
                & np.isfinite(y[safe_target_i])
            )
            rows = anchor_group.index
            output.loc[rows, f"target_observed__{tag}"] = valid
            output.loc[rows[valid], f"target_temp_pl_C__{tag}"] = y[safe_target_i[valid]]
            output.loc[rows[valid], f"target_timestamp__{tag}"] = observed_t[valid]
            output.loc[rows[valid], f"effective_horizon_s__{tag}"] = observed_t[valid] - anchor_t[valid]
    return output
