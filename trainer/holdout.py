"""Cold-start coordinate initializers used by E3-A and E3-B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from client.client import FederatedClient
from data.schema import DataSplit
from server.server import FedTaskPromptServer
from trainer.evaluator import EvaluationSummary, FederatedEvaluator

COLD_START_INITIALIZERS = (
    "zero",
    "group_mean",
    "nearest_task",
    "semantic_generator",
    "oracle_group",
)


@dataclass(frozen=True, slots=True)
class HoldoutPlacement:
    group_id: str
    coordinates: Tensor
    initializer: str


def select_one_client_per_task(
    clients: Mapping[str, FederatedClient],
    *,
    seed: int,
) -> tuple[str, ...]:
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[str]] = {}
    for client_id, client in clients.items():
        by_task.setdefault(client.state.task_id, []).append(client_id)
    selected: list[str] = []
    for _, client_ids in sorted(by_task.items()):
        ordered = sorted(client_ids)
        selected.append(ordered[int(rng.integers(len(ordered)))])
    return tuple(selected)


def clients_for_task(
    clients: Mapping[str, FederatedClient],
    task_id: str,
) -> tuple[str, ...]:
    selected = tuple(
        sorted(
            client_id
            for client_id, client in clients.items()
            if client.state.task_id == task_id
        )
    )
    if not selected:
        raise ValueError(f"no clients found for task {task_id!r}")
    return selected


def _cosine(left: Tensor, right: Tensor) -> float:
    # Task embeddings are intentionally stored on CPU, while GroupState follows
    # the generator device. Similarity is routing metadata and needs no gradient,
    # so compare detached float32 copies on CPU.
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    return float(
        torch.dot(
            F.normalize(left, dim=0),
            F.normalize(right, dim=0),
        )
    )


def place_holdout_client(
    server: FedTaskPromptServer,
    client_id: str,
    *,
    initializer: str,
    training_client_ids: Sequence[str],
) -> HoldoutPlacement:
    if initializer not in COLD_START_INITIALIZERS:
        raise ValueError(f"unknown cold-start initializer {initializer!r}")
    assigned_group = server.client_assignments[client_id]
    semantic = server.initial_coordinates(client_id)
    if initializer == "semantic_generator":
        return HoldoutPlacement(assigned_group, semantic.detach(), initializer)
    if initializer == "zero":
        return HoldoutPlacement(assigned_group, torch.zeros_like(semantic), initializer)
    training_ids = tuple(training_client_ids)
    if initializer == "group_mean":
        members = [
            member_id
            for member_id in training_ids
            if server.client_assignments.get(member_id) == assigned_group
        ]
        if not members:
            return HoldoutPlacement(assigned_group, torch.zeros_like(semantic), initializer)
        stacked = torch.stack(
            [server.initial_coordinates(member_id).detach() for member_id in members]
        )
        return HoldoutPlacement(assigned_group, stacked.mean(dim=0), initializer)
    if initializer == "nearest_task":
        if not training_ids:
            raise ValueError("nearest_task requires at least one training client")
        query = server.task_embeddings[client_id]
        neighbor_id = max(
            training_ids,
            key=lambda member_id: (
                _cosine(query, server.task_embeddings[member_id]),
                member_id,
            ),
        )
        neighbor_group = server.client_assignments[neighbor_id]
        return HoldoutPlacement(
            neighbor_group,
            server.initial_coordinates(neighbor_id).detach(),
            initializer,
        )
    scores = {
        group_id: _cosine(server.task_embeddings[client_id], group.semantic_prototype)
        for group_id, group in server.groups.items()
    }
    oracle_group = max(sorted(scores), key=scores.__getitem__)
    coordinates = server.generator_trainer.initial_coordinates(
        server.task_embeddings[client_id],
        server.groups[oracle_group],
    )
    return HoldoutPlacement(oracle_group, coordinates.detach(), initializer)


def evaluate_initializer_curve(
    *,
    server: FedTaskPromptServer,
    evaluator: FederatedEvaluator,
    clients: Mapping[str, FederatedClient],
    holdout_ids: Sequence[str],
    training_client_ids: Sequence[str],
    initializer: str,
    adaptation_steps: Sequence[int],
    seed: int,
    split: DataSplit = DataSplit.TEST,
) -> list[EvaluationSummary]:
    placements = {
        client_id: place_holdout_client(
            server,
            client_id,
            initializer=initializer,
            training_client_ids=training_client_ids,
        )
        for client_id in holdout_ids
    }
    return [
        evaluator.evaluate(
            server=server,
            clients=clients,
            client_ids=tuple(holdout_ids),
            adaptation_steps=int(steps),
            seed=seed,
            split=split,
            coordinate_fn=lambda client_id, _placements=placements: _placements[
                client_id
            ].coordinates,
            group_id_fn=lambda client_id, _placements=placements: _placements[
                client_id
            ].group_id,
        )
        for steps in adaptation_steps
    ]
