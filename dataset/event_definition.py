"""Deterministic event-episode definitions for RAISE-FPGA.

This module deliberately contains no threshold-selection logic.  A physical-unit
threshold must be frozen from development/training data before this code is run
on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventDefinition:
    """Frozen one-sided, sustained high-temperature episode definition."""

    threshold_c: float
    persistence_s: float
    recovery_threshold_c: float
    recovery_s: float
    refractory_s: float
    max_sample_gap_s: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold_c):
            raise ValueError("threshold_c must be finite")
        if not np.isfinite(self.recovery_threshold_c):
            raise ValueError("recovery_threshold_c must be finite")
        if self.recovery_threshold_c >= self.threshold_c:
            raise ValueError("recovery threshold must be below onset threshold")
        for name in (
            "persistence_s",
            "recovery_s",
            "refractory_s",
            "max_sample_gap_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RangeEventDefinition:
    """Frozen sustained excursion outside a two-sided operating range."""

    event_lower: float
    event_upper: float
    recovery_lower: float
    recovery_upper: float
    persistence_s: float
    recovery_s: float
    refractory_s: float
    max_sample_gap_s: float

    def __post_init__(self) -> None:
        bounds = (
            self.event_lower,
            self.event_upper,
            self.recovery_lower,
            self.recovery_upper,
        )
        if not all(np.isfinite(x) for x in bounds):
            raise ValueError("range bounds must be finite")
        if not self.event_lower < self.recovery_lower:
            raise ValueError("recovery lower bound must be inside event lower bound")
        if not self.recovery_lower < self.recovery_upper:
            raise ValueError("recovery lower bound must be below recovery upper")
        if not self.recovery_upper < self.event_upper:
            raise ValueError("recovery upper bound must be inside event upper bound")
        for name in (
            "persistence_s",
            "recovery_s",
            "refractory_s",
            "max_sample_gap_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


EPISODE_COLUMNS = [
    "run_id",
    "episode_id",
    "onset_timestamp",
    "confirmation_timestamp",
    "recovery_timestamp",
    "peak_temperature_c",
    "duration_s",
    "left_censored",
    "right_censored",
]


def _seconds(delta: object) -> float:
    """Convert a numeric or pandas-like time delta to seconds."""

    if isinstance(delta, pd.Timedelta):
        return float(delta.total_seconds())
    if isinstance(delta, np.timedelta64):
        return float(delta / np.timedelta64(1, "s"))
    return float(delta)


def detect_temperature_episodes(
    frame: pd.DataFrame,
    *,
    run_col: str,
    timestamp_col: str,
    temperature_col: str,
    definition: EventDefinition,
) -> pd.DataFrame:
    """Detect independent sustained high-temperature event episodes.

    The onset timestamp is the first above-threshold observation in a continuous
    excursion that is later confirmed by ``persistence_s`` of observation.  A
    gap larger than ``max_sample_gap_s`` breaks continuity and never contributes
    evidence for onset, persistence, or recovery.  An episode ends only after a
    continuous recovery excursion at or below ``recovery_threshold_c``.  Starts
    found at a run's first valid observation are flagged as left-censored rather
    than silently treated as confirmed new onsets.

    Refractory time is measured from recovery.  A qualifying excursion beginning
    during that interval is merged with the prior episode.  NaN temperature or
    timestamp rows are excluded and duplicate timestamps keep the last value;
    callers should separately audit duplicate consistency before using this
    function.
    """

    required = {run_col, timestamp_col, temperature_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    episodes: list[dict[str, object]] = []
    for run_id, group in frame.groupby(run_col, sort=False, dropna=False):
        g = group[[timestamp_col, temperature_col]].dropna().copy()
        g = g.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last")
        if g.empty:
            continue

        timestamps = g[timestamp_col].to_numpy()
        values = g[temperature_col].astype(float).to_numpy()
        candidate_i: int | None = None
        active: dict[str, object] | None = None
        recovery_i: int | None = None
        prior_timestamp: object | None = None
        last_recovery_timestamp: object | None = None

        for i, (timestamp, value) in enumerate(zip(timestamps, values, strict=True)):
            discontinuity = (
                prior_timestamp is not None
                and _seconds(timestamp - prior_timestamp) > definition.max_sample_gap_s
            )
            if discontinuity:
                candidate_i = None
                recovery_i = None
                # An active episode is right-censored by the gap.  It cannot
                # establish recovery, so close it without asserting a recovery.
                if active is not None:
                    active["recovery_timestamp"] = pd.NaT
                    active["duration_s"] = np.nan
                    active["right_censored"] = True
                    episodes.append(active)
                    active = None

            if active is None:
                if value >= definition.threshold_c:
                    if candidate_i is None:
                        candidate_i = i
                    elapsed = _seconds(timestamp - timestamps[candidate_i])
                    if elapsed + 1e-12 >= definition.persistence_s:
                        onset = timestamps[candidate_i]
                        within_refractory = (
                            last_recovery_timestamp is not None
                            and _seconds(onset - last_recovery_timestamp)
                            < definition.refractory_s
                        )
                        if within_refractory:
                            # Reopen the just-closed episode: this excursion is
                            # not an independent event under the frozen
                            # separation rule.
                            active = episodes.pop()
                            active["recovery_timestamp"] = pd.NaT
                            active["duration_s"] = np.nan
                            active["right_censored"] = True
                            active["peak_temperature_c"] = max(
                                float(active["peak_temperature_c"]),
                                float(np.max(values[candidate_i : i + 1])),
                            )
                            last_recovery_timestamp = None
                        else:
                            active = {
                                "run_id": run_id,
                                "episode_id": None,
                                "onset_timestamp": onset,
                                "confirmation_timestamp": timestamp,
                                "recovery_timestamp": pd.NaT,
                                "peak_temperature_c": float(
                                    np.max(values[candidate_i : i + 1])
                                ),
                                "duration_s": np.nan,
                                "left_censored": bool(candidate_i == 0),
                                "right_censored": True,
                            }
                        candidate_i = None
                else:
                    candidate_i = None
            else:
                active["peak_temperature_c"] = max(
                    float(active["peak_temperature_c"]), float(value)
                )
                if value <= definition.recovery_threshold_c:
                    if recovery_i is None:
                        recovery_i = i
                    if (
                        _seconds(timestamp - timestamps[recovery_i]) + 1e-12
                        >= definition.recovery_s
                    ):
                        recovery_timestamp = timestamps[recovery_i]
                        active["recovery_timestamp"] = recovery_timestamp
                        active["duration_s"] = _seconds(
                            recovery_timestamp - active["onset_timestamp"]
                        )
                        active["right_censored"] = False
                        episodes.append(active)
                        active = None
                        recovery_i = None
                        last_recovery_timestamp = recovery_timestamp
                else:
                    recovery_i = None

            prior_timestamp = timestamp

        if active is not None:
            episodes.append(active)

    result = pd.DataFrame(episodes, columns=EPISODE_COLUMNS)
    if not result.empty:
        result["episode_id"] = (
            result.groupby("run_id", sort=False).cumcount().add(1).astype(int)
        )
    return result


def confirmed_onsets(episodes: pd.DataFrame) -> pd.DataFrame:
    """Return episodes eligible as new-onset forecasting targets."""

    if "left_censored" not in episodes:
        raise KeyError("episodes must contain left_censored")
    return episodes.loc[~episodes["left_censored"].astype(bool)].copy()


def count_onsets_by_run(
    episodes: pd.DataFrame,
    *,
    workload_by_run: dict[object, object] | None = None,
) -> pd.DataFrame:
    """Count confirmed independent onsets by run for feasibility reporting."""

    eligible = confirmed_onsets(episodes)
    counts = eligible.groupby("run_id", dropna=False).size().rename("event_onsets")
    result = counts.reset_index()
    if workload_by_run is not None:
        result["workload_group"] = result["run_id"].map(workload_by_run)
    return result


RANGE_EPISODE_COLUMNS = [
    "run_id",
    "episode_id",
    "onset_timestamp",
    "confirmation_timestamp",
    "recovery_timestamp",
    "minimum_value",
    "maximum_value",
    "duration_s",
    "left_censored",
    "right_censored",
]


def detect_range_episodes(
    frame: pd.DataFrame,
    *,
    run_col: str,
    timestamp_col: str,
    value_col: str,
    definition: RangeEventDefinition,
) -> pd.DataFrame:
    """Detect sustained excursions outside a frozen two-sided range.

    Event state is entered below ``event_lower`` or above ``event_upper`` and is
    cleared only after continuous observation inside the narrower recovery
    range.  Gap, censoring, timestamp, and refractory semantics match
    :func:`detect_temperature_episodes`.
    """

    required = {run_col, timestamp_col, value_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    episodes: list[dict[str, object]] = []
    for run_id, group in frame.groupby(run_col, sort=False, dropna=False):
        g = group[[timestamp_col, value_col]].dropna().copy()
        g = g.sort_values(timestamp_col).drop_duplicates(timestamp_col, keep="last")
        if g.empty:
            continue

        timestamps = g[timestamp_col].to_numpy()
        values = g[value_col].astype(float).to_numpy()
        candidate_i: int | None = None
        active: dict[str, object] | None = None
        recovery_i: int | None = None
        prior_timestamp: object | None = None
        last_recovery_timestamp: object | None = None

        for i, (timestamp, value) in enumerate(zip(timestamps, values, strict=True)):
            discontinuity = (
                prior_timestamp is not None
                and _seconds(timestamp - prior_timestamp) > definition.max_sample_gap_s
            )
            if discontinuity:
                candidate_i = None
                recovery_i = None
                if active is not None:
                    active["recovery_timestamp"] = pd.NaT
                    active["duration_s"] = np.nan
                    active["right_censored"] = True
                    episodes.append(active)
                    active = None

            outside = value < definition.event_lower or value > definition.event_upper
            recovered = (
                definition.recovery_lower <= value <= definition.recovery_upper
            )
            if active is None:
                if outside:
                    if candidate_i is None:
                        candidate_i = i
                    elapsed = _seconds(timestamp - timestamps[candidate_i])
                    if elapsed + 1e-12 >= definition.persistence_s:
                        onset = timestamps[candidate_i]
                        within_refractory = (
                            last_recovery_timestamp is not None
                            and _seconds(onset - last_recovery_timestamp)
                            < definition.refractory_s
                        )
                        if within_refractory:
                            active = episodes.pop()
                            active["recovery_timestamp"] = pd.NaT
                            active["duration_s"] = np.nan
                            active["right_censored"] = True
                            active["minimum_value"] = min(
                                float(active["minimum_value"]),
                                float(np.min(values[candidate_i : i + 1])),
                            )
                            active["maximum_value"] = max(
                                float(active["maximum_value"]),
                                float(np.max(values[candidate_i : i + 1])),
                            )
                            last_recovery_timestamp = None
                        else:
                            active = {
                                "run_id": run_id,
                                "episode_id": None,
                                "onset_timestamp": onset,
                                "confirmation_timestamp": timestamp,
                                "recovery_timestamp": pd.NaT,
                                "minimum_value": float(
                                    np.min(values[candidate_i : i + 1])
                                ),
                                "maximum_value": float(
                                    np.max(values[candidate_i : i + 1])
                                ),
                                "duration_s": np.nan,
                                "left_censored": bool(candidate_i == 0),
                                "right_censored": True,
                            }
                        candidate_i = None
                else:
                    candidate_i = None
            else:
                active["minimum_value"] = min(
                    float(active["minimum_value"]), float(value)
                )
                active["maximum_value"] = max(
                    float(active["maximum_value"]), float(value)
                )
                if recovered:
                    if recovery_i is None:
                        recovery_i = i
                    if (
                        _seconds(timestamp - timestamps[recovery_i]) + 1e-12
                        >= definition.recovery_s
                    ):
                        recovery_timestamp = timestamps[recovery_i]
                        active["recovery_timestamp"] = recovery_timestamp
                        active["duration_s"] = _seconds(
                            recovery_timestamp - active["onset_timestamp"]
                        )
                        active["right_censored"] = False
                        episodes.append(active)
                        active = None
                        recovery_i = None
                        last_recovery_timestamp = recovery_timestamp
                else:
                    recovery_i = None

            prior_timestamp = timestamp

        if active is not None:
            episodes.append(active)

    result = pd.DataFrame(episodes, columns=RANGE_EPISODE_COLUMNS)
    if not result.empty:
        result["episode_id"] = (
            result.groupby("run_id", sort=False).cumcount().add(1).astype(int)
        )
    return result
