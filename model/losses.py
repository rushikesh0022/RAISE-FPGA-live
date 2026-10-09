"""Numerically stable discrete-time hazard likelihood."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def discrete_hazard_nll(
    logits: torch.Tensor,
    event_interval: torch.Tensor,
    observed_intervals: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return mean NLL for events (0..2) or event-free/censored samples (-1).

    ``observed_intervals`` is the number of intervals with valid follow-up for
    an event-free/censored sample (0..3). Event rows must observe through their
    event interval. Fully observed event-free rows use 3.
    """

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("logits must have shape [batch, 3]")
    event_interval = event_interval.to(device=logits.device, dtype=torch.long)
    if observed_intervals is None:
        observed_intervals = torch.full_like(event_interval, 3)
    else:
        observed_intervals = observed_intervals.to(device=logits.device, dtype=torch.long)
    if torch.any((event_interval < -1) | (event_interval > 2)):
        raise ValueError("event_interval values must be -1, 0, 1, or 2")
    if torch.any((observed_intervals < 0) | (observed_intervals > 3)):
        raise ValueError("observed_intervals values must be in 0..3")

    event_rows = event_interval >= 0
    if torch.any(event_rows & (observed_intervals <= event_interval)):
        raise ValueError("event interval is not observed")

    log_survival = functional.logsigmoid(-logits)
    log_hazard = functional.logsigmoid(logits)
    interval_index = torch.arange(3, device=logits.device).unsqueeze(0)
    # Event rows survive only the intervals before the event. Censored/no-event
    # rows contribute every interval for which follow-up is observed.
    survival_limit = torch.where(event_rows, event_interval, observed_intervals)
    losses = -(log_survival * (interval_index < survival_limit.unsqueeze(1))).sum(dim=1)
    safe_event_index = event_interval.clamp_min(0).unsqueeze(1)
    event_term = log_hazard.gather(1, safe_event_index).squeeze(1)
    losses = losses - torch.where(event_rows, event_term, torch.zeros_like(event_term))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")
