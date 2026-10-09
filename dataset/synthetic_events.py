"""Synthetic, internally consistent thermal-transition histories for model research."""

from __future__ import annotations

import numpy as np


THRESHOLDS_C = np.asarray((49.0, 50.0, 51.0), dtype=np.float32)
HORIZONS_S = np.asarray((0.5, 2.0, 5.0), dtype=np.float32)


def generate_synthetic_events(
    samples: int,
    *,
    seed: int,
    history_samples: int = 33,
    cadence_hz: float = 10.0,
) -> dict[str, np.ndarray]:
    """Generate balanced ramp/plateau histories and exact future onset labels.

    A focus threshold and one of four outcomes (B1/B2/B3/no event) are sampled
    uniformly. Other threshold labels are derived from the same latent future,
    preserving nested physical ordering rather than being invented independently.
    """

    rng = np.random.default_rng(seed)
    focus = rng.integers(0, 3, samples)
    outcome = rng.integers(0, 4, samples)
    rate = rng.uniform(0.18, 1.25, samples).astype(np.float32)
    crossing = np.empty(samples, np.float32)
    ranges = ((0.08, 0.48), (0.55, 1.95), (2.05, 4.95), (5.4, 8.0))
    for interval, (lower, upper) in enumerate(ranges):
        selected = outcome == interval
        crossing[selected] = rng.uniform(lower, upper, selected.sum())
    current = THRESHOLDS_C[focus] - rate * crossing
    current += rng.normal(0.0, 0.015, samples).astype(np.float32)

    age = np.arange(history_samples - 1, -1, -1, dtype=np.float32) / cadence_hz
    thermal_lag = rng.uniform(0.0, 0.06, (samples, 1)).astype(np.float32)
    pl = current[:, None] - rate[:, None] * age[None, :] + thermal_lag
    pl += rng.normal(0.0, 0.012, pl.shape).astype(np.float32)
    ps = pl - rng.uniform(0.4, 1.0, (samples, 1)).astype(np.float32)
    remote = pl - rng.uniform(0.8, 1.6, (samples, 1)).astype(np.float32)
    power = 4.0 + 2.8 * rate[:, None] + 0.03 * pl
    power += rng.normal(0.0, 0.025, pl.shape).astype(np.float32)
    vccint = 0.85 - 0.006 * rate[:, None] + rng.normal(0, 0.001, pl.shape)
    vccaux = 1.80 - 0.003 * rate[:, None] + rng.normal(0, 0.001, pl.shape)
    vccbram = 0.95 - 0.004 * rate[:, None] + rng.normal(0, 0.001, pl.shape)
    history = np.stack((pl, ps, remote, vccint, vccaux, vccbram, power), axis=-1)
    history = history.astype(np.float32)

    event_interval = np.full((samples, 3), -1, dtype=np.int64)
    observed_intervals = np.full((samples, 3), 3, dtype=np.int64)
    eligible = current[:, None] < THRESHOLDS_C[None, :]
    for severity, threshold in enumerate(THRESHOLDS_C):
        time_to_cross = (threshold - current) / rate
        interval = np.searchsorted(HORIZONS_S, time_to_cross, side="left")
        event_interval[:, severity] = np.where(
            eligible[:, severity] & (interval < 3), interval, -1
        )
        observed_intervals[:, severity] = np.where(eligible[:, severity], 3, 0)
    return {
        "history": history,
        "event_interval": event_interval,
        "observed_intervals": observed_intervals,
        "current_temperature_c": current.astype(np.float32),
        "heating_rate_c_per_s": rate,
        "thresholds_c": THRESHOLDS_C.copy(),
        "horizons_s": HORIZONS_S.copy(),
    }
