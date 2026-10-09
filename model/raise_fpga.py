"""Compact multi-scale cumulative-hazard network from the governing prompt."""

from __future__ import annotations

import torch
from torch import nn


class LatestDepthwiseCausalFilter(nn.Module):
    """One three-tap depthwise filter evaluated only at the latest time."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(channels, 3))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.channels:
            raise ValueError("history must have shape [batch, time, channels]")
        required = 2 * self.dilation + 1
        if history.shape[1] < required:
            raise ValueError(f"history needs at least {required} samples")
        # With the latest sample at index -1, the three causal taps are
        # t-2d, t-d, and t (not Python's -2d, -d, -1 positions).
        indices = torch.tensor(
            [-(2 * self.dilation + 1), -(self.dilation + 1), -1],
            device=history.device,
        )
        taps = history.index_select(1, indices % history.shape[1])
        return torch.relu((taps * self.weight.T.unsqueeze(0)).sum(dim=1) + self.bias)


class RaiseEncoder(nn.Module):
    """Shared temporal and low-rank state encoder."""

    def __init__(
        self,
        channels: int = 8,
        dilations: tuple[int, int, int] = (1, 4, 16),
        interaction_rank: int = 8,
        state_dimension: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.filters = nn.ModuleList(
            LatestDepthwiseCausalFilter(channels, dilation) for dilation in dilations
        )
        z_dimension = channels * (len(dilations) + 1)
        self.cross_p = nn.Linear(z_dimension, interaction_rank)
        self.cross_q = nn.Linear(z_dimension, interaction_rank)
        self.state = nn.Linear(z_dimension + interaction_rank, state_dimension)
        self.state_dimension = state_dimension

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        temporal = [branch(history) for branch in self.filters]
        current = history[:, -1, :]
        z = torch.cat([*temporal, current], dim=-1)
        interaction = self.cross_p(z) * self.cross_q(z)
        return torch.relu(self.state(torch.cat([z, interaction], dim=-1)))


class RaiseFPGA(nn.Module):
    """Single-event risk model; action selection is deterministic and separate."""

    def __init__(
        self,
        channels: int = 8,
        dilations: tuple[int, int, int] = (1, 4, 16),
        interaction_rank: int = 8,
        state_dimension: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = RaiseEncoder(
            channels, dilations, interaction_rank, state_dimension
        )
        self.hazard_logits = nn.Linear(state_dimension, 3)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(history)
        logits = self.hazard_logits(state)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=-1)
        cumulative_risk = 1.0 - survival
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "cumulative_risk": cumulative_risk,
            "state": state,
        }


class MultiEventRaiseFPGA(nn.Module):
    """Explicit event-specific hazard heads sharing one compact encoder."""

    def __init__(
        self,
        event_types: tuple[str, ...],
        channels: int = 8,
        dilations: tuple[int, int, int] = (1, 4, 16),
        interaction_rank: int = 8,
        state_dimension: int = 16,
    ) -> None:
        super().__init__()
        if not event_types or len(set(event_types)) != len(event_types):
            raise ValueError("event_types must be non-empty and unique")
        self.event_types = event_types
        self.encoder = RaiseEncoder(
            channels, dilations, interaction_rank, state_dimension
        )
        self.hazard_logits = nn.Linear(state_dimension, len(event_types) * 3)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(history)
        logits = self.hazard_logits(state).reshape(-1, len(self.event_types), 3)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "cumulative_risk": cumulative_risk,
            "state": state,
        }


class RaiseFPGAWithTemperatureAux(nn.Module):
    """Single-event hazard model with dense future-temperature-delta auxiliary task."""

    def __init__(
        self,
        channels: int = 7,
        dilations: tuple[int, int, int] = (1, 4, 16),
        interaction_rank: int = 8,
        state_dimension: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = RaiseEncoder(
            channels, dilations, interaction_rank, state_dimension
        )
        self.hazard_logits = nn.Linear(state_dimension, 3)
        self.future_temperature_delta = nn.Linear(state_dimension, 3)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(history)
        logits = self.hazard_logits(state)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "cumulative_risk": cumulative_risk,
            "future_temperature_delta": self.future_temperature_delta(state),
            "state": state,
        }


class NestedSeverityRaiseFPGA(nn.Module):
    """Ordinal temperature-zone hazards with a dense forecasting auxiliary task.

    The three severity outputs are WATCH, ELEVATED, and HIGH.  Non-negative
    logit decrements make every interval hazard no larger for a more severe
    zone.  Combined with cumulative interval hazards this guarantees both
    severity ordering and horizon ordering by construction.
    """

    severity_names = ("watch", "elevated", "high")

    def __init__(
        self,
        channels: int = 7,
        dilations: tuple[int, int, int] = (1, 4, 16),
        interaction_rank: int = 8,
        state_dimension: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = RaiseEncoder(
            channels, dilations, interaction_rank, state_dimension
        )
        self.watch_hazard_logits = nn.Linear(state_dimension, 3)
        self.severity_logit_decrements = nn.Linear(state_dimension, 6)
        self.future_temperature_delta = nn.Linear(state_dimension, 3)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.encoder(history)
        watch = self.watch_hazard_logits(state)
        decrements = torch.nn.functional.softplus(
            self.severity_logit_decrements(state).reshape(-1, 2, 3)
        )
        elevated = watch - decrements[:, 0]
        high = elevated - decrements[:, 1]
        logits = torch.stack((watch, elevated, high), dim=1)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "cumulative_risk": cumulative_risk,
            "future_temperature_delta": self.future_temperature_delta(state),
            "state": state,
        }


class PhysicsGuidedNestedRaiseFPGA(nn.Module):
    """Fused multi-scale, recurrent, and explicit thermal-trend encoder."""

    severity_names = ("transition_49c", "transition_50c", "transition_51c")

    def __init__(
        self,
        channels: int = 7,
        temperature_channel: int = 0,
        recurrent_dimension: int = 24,
        fused_dimension: int = 32,
    ) -> None:
        super().__init__()
        self.temperature_channel = temperature_channel
        self.multiscale = RaiseEncoder(channels=channels, state_dimension=16)
        self.recurrent = nn.GRU(
            input_size=channels,
            hidden_size=recurrent_dimension,
            batch_first=True,
        )
        self.trend = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(16 + recurrent_dimension + 16, fused_dimension),
            nn.ReLU(),
        )
        self.watch_hazard_logits = nn.Linear(fused_dimension, 3)
        self.severity_logit_decrements = nn.Linear(fused_dimension, 6)
        self.future_temperature_delta = nn.Linear(fused_dimension, 3)

    def _trend_features(self, history: torch.Tensor) -> torch.Tensor:
        temperature = history[:, :, self.temperature_channel]
        current = temperature[:, -1]
        short = current - temperature[:, -6]
        medium = current - temperature[:, -21]
        long = current - temperature[:, 0]
        acceleration = short - (temperature[:, -6] - temperature[:, -11])
        variability = temperature.std(dim=1, unbiased=False)
        return torch.stack(
            (current, short, medium, long, acceleration, variability), dim=-1
        )

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        multiscale_state = self.multiscale(history)
        _, recurrent_state = self.recurrent(history)
        trend_state = self.trend(self._trend_features(history))
        state = self.fusion(
            torch.cat((multiscale_state, recurrent_state[-1], trend_state), dim=-1)
        )
        watch = self.watch_hazard_logits(state)
        decrements = torch.nn.functional.softplus(
            self.severity_logit_decrements(state).reshape(-1, 2, 3)
        )
        elevated = watch - decrements[:, 0]
        high = elevated - decrements[:, 1]
        logits = torch.stack((watch, elevated, high), dim=1)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "cumulative_risk": cumulative_risk,
            "future_temperature_delta": self.future_temperature_delta(state),
            "state": state,
        }


class TrajectoryConditionedRaiseFPGA(nn.Module):
    """Ordered hazards conditioned by a dense thermal-trajectory encoder.

    The required 1/4/16 branches and rank-8 interaction are retained.  A
    depthwise residual TCN adds capacity without recurrent hardware, while a
    dense 0.1-second temperature decoder supplies training signal at many more
    anchors than the rare onset labels.  The decoder may be pruned from a
    risk-only FPGA export after joint training.
    """

    severity_names = ("transition_49c", "transition_50c", "transition_51c")

    def __init__(
        self,
        channels: int = 7,
        temperature_channel: int = 0,
        hidden_dimension: int = 32,
        state_dimension: int = 48,
        trajectory_steps: int = 50,
    ) -> None:
        super().__init__()
        self.temperature_channel = temperature_channel
        self.trajectory_steps = trajectory_steps
        self.required_encoder = RaiseEncoder(
            channels=channels,
            dilations=(1, 4, 16),
            interaction_rank=8,
            state_dimension=16,
        )
        # Fixed nested causal patches are cheaper than a full sequence model on
        # both CPU and FPGA: mean, spread, and endpoint change at four scales.
        self.patch_widths = (5, 11, 21, 33)
        self.patch_projection = nn.Sequential(
            nn.Linear(channels * 3 * len(self.patch_widths), hidden_dimension),
            nn.ReLU(),
        )
        self.trend = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(16 + hidden_dimension + 16, state_dimension),
            nn.ReLU(),
            nn.Linear(state_dimension, state_dimension),
            nn.ReLU(),
        )
        self.watch_hazard_logits = nn.Linear(state_dimension, 3)
        self.severity_logit_decrements = nn.Linear(state_dimension, 6)
        self.trajectory_delta = nn.Linear(state_dimension, trajectory_steps)
        self.trajectory_log_scale = nn.Linear(state_dimension, 3)

    def _trend_features(self, history: torch.Tensor) -> torch.Tensor:
        temperature = history[:, :, self.temperature_channel]
        current = temperature[:, -1]
        short = current - temperature[:, -6]
        medium = current - temperature[:, -21]
        long = current - temperature[:, 0]
        acceleration = short - (temperature[:, -6] - temperature[:, -11])
        variability = temperature.std(dim=1, unbiased=False)
        return torch.stack(
            (
                current,
                short,
                medium,
                long,
                acceleration,
                variability,
                temperature.amin(dim=1),
                temperature.amax(dim=1),
            ),
            dim=-1,
        )

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        required_state = self.required_encoder(history)
        summaries = []
        for width in self.patch_widths:
            patch = history[:, -width:, :]
            summaries.extend(
                (
                    patch.mean(dim=1),
                    patch.std(dim=1, unbiased=False),
                    patch[:, -1, :] - patch[:, 0, :],
                )
            )
        temporal_state = self.patch_projection(torch.cat(summaries, dim=-1))
        trend_state = self.trend(self._trend_features(history))
        state = self.fusion(
            torch.cat((required_state, temporal_state, trend_state), dim=-1)
        )
        watch = self.watch_hazard_logits(state)
        decrements = torch.nn.functional.softplus(
            self.severity_logit_decrements(state).reshape(-1, 2, 3)
        )
        elevated = watch - decrements[:, 0]
        high = elevated - decrements[:, 1]
        logits = torch.stack((watch, elevated, high), dim=1)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)

        trajectory = self.trajectory_delta(state)
        horizon_indices = torch.tensor(
            (4, 19, 49), dtype=torch.long, device=trajectory.device
        )
        return {
            "hazard_logits": logits,
            "hazards": hazards,
            "cumulative_risk": cumulative_risk,
            "future_temperature_delta": trajectory.index_select(1, horizon_indices),
            "temperature_trajectory_delta": trajectory,
            "temperature_trajectory_scale": torch.nn.functional.softplus(
                self.trajectory_log_scale(state)
            )
            + 1e-4,
            "state": state,
        }


class FutureRegionOccupancyRaiseFPGA(nn.Module):
    """Class-balanced future-region head over the trajectory representation."""

    severity_names = TrajectoryConditionedRaiseFPGA.severity_names

    def __init__(self, channels: int = 7) -> None:
        super().__init__()
        self.backbone = TrajectoryConditionedRaiseFPGA(channels=channels)
        self.watch_occupancy_logits = nn.Linear(48, 3)
        self.severity_logit_decrements = nn.Linear(48, 6)

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(history)
        state = backbone_output["state"]
        watch = self.watch_occupancy_logits(state)
        decrements = torch.nn.functional.softplus(
            self.severity_logit_decrements(state).reshape(-1, 2, 3)
        )
        elevated = watch - decrements[:, 0]
        high = elevated - decrements[:, 1]
        logits = torch.stack((watch, elevated, high), dim=1)
        return {
            **backbone_output,
            "occupancy_logits": logits,
            "occupancy_probability": torch.sigmoid(logits),
        }
