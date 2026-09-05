"""Sequential client evaluation for the single-backbone prototype."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from client.client import FederatedClient
from client.state import ClientEvaluationResult
from data.schema import DataSplit
from server.server import FedTaskPromptServer


@dataclass(frozen=True, slots=True)
class TaskMetricSummary:
    task_id: str
    metric_name: str
    value: float
    num_examples: int


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    round_number: int
    adaptation_steps: int
    split: DataSplit
    client_results: tuple[ClientEvaluationResult, ...]
    mean_test_loss: float
    client_std_test_loss: float
    worst_fraction_test_loss: float
    task_metrics: tuple[TaskMetricSummary, ...]


def summarize_client_results(
    results: Sequence[ClientEvaluationResult],
    *,
    clients: Mapping[str, FederatedClient],
    round_number: int,
    adaptation_steps: int,
    split: DataSplit,
    worst_client_fraction: float,
) -> EvaluationSummary:
    values = tuple(results)
    if not values:
        raise ValueError("evaluation results cannot be empty")
    weights = np.asarray([result.test_examples for result in values], dtype=np.float64)
    weights /= weights.sum()
    losses = np.asarray([result.test_loss for result in values], dtype=np.float64)
    mean_loss = float(np.dot(weights, losses))
    std_loss = float(np.sqrt(np.dot(weights, (losses - mean_loss) ** 2)))
    worst_count = max(1, math.ceil(len(values) * worst_client_fraction))
    worst_loss = float(np.sort(losses)[-worst_count:].mean())
    task_metrics: list[TaskMetricSummary] = []
    task_ids = sorted({clients[result.client_id].state.task_id for result in values})
    for task_id in task_ids:
        task_results = [
            result
            for result in values
            if clients[result.client_id].state.task_id == task_id
        ]
        metric_results = [
            result
            for result in task_results
            if result.metric_name is not None and result.metric_value is not None
        ]
        if not metric_results:
            continue
        if len(metric_results) != len(task_results):
            raise ValueError(f"task {task_id!r} has incomplete metric results")
        metric_names = {result.metric_name for result in metric_results}
        if len(metric_names) != 1:
            raise ValueError(f"task {task_id!r} has inconsistent metric names")
        counts = np.asarray(
            [result.test_examples for result in metric_results],
            dtype=np.float64,
        )
        metric_values = np.asarray(
            [result.metric_value for result in metric_results],
            dtype=np.float64,
        )
        task_metrics.append(
            TaskMetricSummary(
                task_id=task_id,
                metric_name=str(metric_results[0].metric_name),
                value=float(np.dot(counts / counts.sum(), metric_values)),
                num_examples=int(counts.sum()),
            )
        )
    return EvaluationSummary(
        round_number=round_number,
        adaptation_steps=adaptation_steps,
        split=split,
        client_results=values,
        mean_test_loss=mean_loss,
        client_std_test_loss=std_loss,
        worst_fraction_test_loss=worst_loss,
        task_metrics=tuple(task_metrics),
    )


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
        split: DataSplit = DataSplit.VALIDATION,
        coordinate_fn: Callable[[str], Tensor] | None = None,
        group_id_fn: Callable[[str], str] | None = None,
    ) -> EvaluationSummary:
        selected = tuple(sorted(clients) if client_ids is None else client_ids)
        if not selected:
            raise ValueError("evaluation client set cannot be empty")
        results: list[ClientEvaluationResult] = []
        for position, client_id in enumerate(selected):
            if client_id not in clients:
                raise KeyError(f"unknown evaluation client {client_id!r}")
            group_id = (
                group_id_fn(client_id)
                if group_id_fn is not None
                else server.client_assignments[client_id]
            )
            group = server.groups[group_id]
            coordinates = (
                coordinate_fn(client_id)
                if coordinate_fn is not None
                else server.initial_coordinates(client_id)
            )
            results.append(
                clients[client_id].evaluate_loss(
                    group_id=group_id,
                    initial_coordinates=coordinates,
                    subspace=group.subspace,
                    config=server.inner_loop,
                    adaptation_steps=adaptation_steps,
                    seed=seed + position,
                    split=split,
                )
            )
        return summarize_client_results(
            results,
            clients=clients,
            round_number=server.round_number,
            adaptation_steps=adaptation_steps,
            split=split,
            worst_client_fraction=self.worst_client_fraction,
        )
