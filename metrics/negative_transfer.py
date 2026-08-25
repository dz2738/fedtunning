"""Client-level gains, negative transfer, fairness, and tail metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from metrics.transfer import relative_gain, signed_gain


@dataclass(frozen=True, slots=True)
class TransferSummary:
    client_ids: tuple[str, ...]
    absolute_gains: Mapping[str, float]
    relative_gains: Mapping[str, float]
    mean_absolute_gain: float
    mean_relative_gain: float
    negative_transfer_rate: float
    method_std: float
    worst_fraction_score: float


def summarize_client_transfer(
    method_scores: Mapping[str, float],
    local_scores: Mapping[str, float],
    *,
    higher_is_better: bool = True,
    negative_tolerance: float = 0.0,
    worst_client_fraction: float = 0.1,
) -> TransferSummary:
    if set(method_scores) != set(local_scores) or not method_scores:
        raise ValueError("method and local scores must contain the same non-empty clients")
    if negative_tolerance < 0:
        raise ValueError("negative_tolerance must be non-negative")
    if not 0.0 < worst_client_fraction <= 1.0:
        raise ValueError("worst_client_fraction must be in (0, 1]")

    client_ids = tuple(sorted(method_scores))
    absolute = {
        client_id: signed_gain(
            method_scores[client_id],
            local_scores[client_id],
            higher_is_better=higher_is_better,
        )
        for client_id in client_ids
    }
    relative = {
        client_id: relative_gain(
            method_scores[client_id],
            local_scores[client_id],
            higher_is_better=higher_is_better,
        )
        for client_id in client_ids
    }
    gains = np.asarray([absolute[client_id] for client_id in client_ids])
    values = np.asarray([method_scores[client_id] for client_id in client_ids])
    worst_count = max(1, math.ceil(len(values) * worst_client_fraction))
    sorted_values = np.sort(values)
    worst = sorted_values[:worst_count] if higher_is_better else sorted_values[-worst_count:]
    return TransferSummary(
        client_ids=client_ids,
        absolute_gains=absolute,
        relative_gains=relative,
        mean_absolute_gain=float(gains.mean()),
        mean_relative_gain=float(np.mean(tuple(relative.values()))),
        negative_transfer_rate=float(np.mean(gains < -negative_tolerance)),
        method_std=float(values.std()),
        worst_fraction_score=float(worst.mean()),
    )
