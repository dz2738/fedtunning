"""Sequential client evaluation for the single-backbone prototype."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from client.client import FederatedClient
from client.state import ClientEvaluationResult
from server.server import FedTaskPromptServer


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    round_number: int
    adaptation_steps: int
    client_results: tuple[ClientEvaluationResult, ...]
    mean_test_loss: float
    client_std_test_loss: float
    worst_fraction_test_loss: float


class FederatedEvaluator:
    def __init__(self, *, worst_client_fraction: float = 0.1) -> None:
        if not 0.0 < worst_client_fraction <= 1.0:
            raise ValueError("worst_client_fraction must be in (0, 1]")
        self.worst_client_fraction = float(worst_client_fraction)

    def evaluate(
        self,
        *,
        server: FedTaskPromptServer,
        clients: Mapping[str, FederatedClient],
        adaptation_steps: int,
        seed: int,
        client_ids: Sequence[str] | None = None,
    ) -> EvaluationSummary:
        selected = tuple(sorted(clients) if client_ids is None else client_ids)
        if not selected:
            raise ValueError("evaluation client set cannot be empty")
        results: list[ClientEvaluationResult] = []
        for position, client_id in enumerate(selected):
            if client_id not in clients:
                raise KeyError(f"unknown evaluation client {client_id!r}")
            group_id = server.grouper.assignments[client_id]
            group = server.groups[group_id]
            results.append(
                clients[client_id].evaluate_loss(
                    group_id=group_id,
                    initial_coordinates=server.initial_coordinates(client_id),
                    subspace=group.subspace,
                    config=server.inner_loop,
                    adaptation_steps=adaptation_steps,
                    seed=seed + server.round_number * 100_019 + position,
                )
            )

        weights = np.asarray([result.test_examples for result in results], dtype=np.float64)
        weights /= weights.sum()
        losses = np.asarray([result.test_loss for result in results], dtype=np.float64)
        mean_loss = float(np.dot(weights, losses))
        std_loss = float(np.sqrt(np.dot(weights, (losses - mean_loss) ** 2)))
        worst_count = max(1, math.ceil(len(results) * self.worst_client_fraction))
        worst_loss = float(np.sort(losses)[-worst_count:].mean())
        return EvaluationSummary(
            round_number=server.round_number,
            adaptation_steps=adaptation_steps,
            client_results=tuple(results),
            mean_test_loss=mean_loss,
            client_std_test_loss=std_loss,
            worst_fraction_test_loss=worst_loss,
        )
