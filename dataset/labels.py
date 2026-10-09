"""Event-specific, censoring-aware interval labels for onset forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (0.5, 2.0, 5.0)


def _normalise_events(
    event_tables: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(event_tables, Mapping):
        parts = []
        for event_type, table in event_tables.items():
            part = table.copy()
            part["event_type"] = event_type
            parts.append(part)
        events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        events = event_tables.copy()
    required = {"event_type", "run_id", "onset_timestamp", "recovery_timestamp"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"event tables missing columns: {sorted(missing)}")
    if "left_censored" not in events:
        events["left_censored"] = False
    return events.sort_values(["event_type", "run_id", "onset_timestamp"], kind="stable")


def make_interval_labels(
    anchors: pd.DataFrame,
    event_tables: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Make separate discrete-hazard targets for every event type.

    Output is long-form: each anchor/event-type pair is a distinct row.  This
    prevents an implicit OR across temperature, voltage, latency, or any future
    endpoint.  ``observed_b*`` is the loss mask: intervals after an event and
    intervals lacking continuous follow-up are censored.  Anchors inside an
    active episode are excluded entirely from new-onset forecasting.
    """

    required = {"run_id", "anchor_timestamp", "followup_end_timestamp"}
    missing = required.difference(anchors.columns)
    if missing:
        raise KeyError(f"anchors missing columns: {sorted(missing)}")
    horizons = np.asarray(horizons_s, dtype=float)
    if len(horizons) != 3 or np.any(~np.isfinite(horizons)) or np.any(np.diff(horizons) <= 0):
        raise ValueError("exactly three strictly increasing finite horizons are required")
    events = _normalise_events(event_tables)
    event_types = events["event_type"].drop_duplicates().tolist()
    identity_cols = [c for c in ("workload_id", "rep_id") if c in anchors.columns]
    indexed = anchors.reset_index(drop=False).rename(columns={"index": "anchor_index"})
    parts: list[pd.DataFrame] = []
    event_groups = {
        key: group
        for key, group in events.groupby(["event_type", "run_id"], sort=False, dropna=False)
    }

    # Work run-by-run using searchsorted.  Runtime is linear in anchors plus a
    # small event-episode term rather than anchors multiplied by episodes.
    for (run_id, anchor_group) in indexed.groupby("run_id", sort=False, dropna=False):
        now = anchor_group["anchor_timestamp"].to_numpy(dtype=float)
        followup_end = anchor_group["followup_end_timestamp"].to_numpy(dtype=float)
        for event_type in event_types:
            run_events = event_groups.get((event_type, run_id), events.iloc[0:0])
            active = np.zeros(len(anchor_group), dtype=bool)
            for episode in run_events.itertuples(index=False):
                onset = float(episode.onset_timestamp)
                recovery = episode.recovery_timestamp
                active |= (now >= onset) & (
                    True if pd.isna(recovery) else now < float(recovery)
                )

            eligible_onsets = np.sort(run_events.loc[
                ~run_events["left_censored"].astype(bool), "onset_timestamp"
            ].to_numpy(dtype=float))
            future_index = np.searchsorted(eligible_onsets, now, side="right")
            has_future = future_index < len(eligible_onsets)
            next_onset = np.full(len(now), np.inf)
            next_onset[has_future] = eligible_onsets[future_index[has_future]]
            delta = next_onset - now
            interval = np.searchsorted(horizons, delta, side="left")
            event_observed = (
                (~active)
                & has_future
                & (interval < 3)
                & (next_onset <= followup_end + 1e-12)
            )

            targets = np.zeros((len(now), 3), dtype=np.int8)
            targets[np.flatnonzero(event_observed), interval[event_observed]] = 1
            observed = now[:, None] + horizons[None, :] <= followup_end[:, None] + 1e-12
            for i in range(3):
                # Once the event occurs, later survival intervals are undefined.
                observed[event_observed & (interval < i), i] = False
            observed[active] = False

            out = anchor_group[
                ["anchor_index", "run_id", *identity_cols, "anchor_timestamp"]
            ].copy()
            out["event_type"] = event_type
            out["label_eligible"] = ~active
            out["exclusion_reason"] = np.where(active, "active_current_event", "")
            # Human-facing bins are B1/B2/B3 (1-based); the loss consumes the
            # explicit zero-based index so no implicit subtraction is needed.
            out["event_interval"] = np.where(event_observed, interval + 1, -1)
            out["event_interval_index"] = np.where(event_observed, interval, -1)
            for i in range(3):
                out[f"label_b{i + 1}"] = targets[:, i]
                out[f"observed_b{i + 1}"] = observed[:, i]
            parts.append(out)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
