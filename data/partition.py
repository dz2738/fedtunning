"""Deterministic client partitioning and support/query/validation/test splitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from data.schema import ClientPartition


@dataclass(frozen=True, slots=True)
class SplitRatios:
    support: float
    query: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.support, self.query, self.validation, self.test)
        if any(value <= 0 for value in values):
            raise ValueError("all split ratios must be positive")
        if not np.isclose(sum(values), 1.0):
            raise ValueError(f"split ratios must sum to 1, got {sum(values)}")


def _validate_partition_request(
    num_examples: int,
    num_clients: int,
    min_examples_per_client: int,
) -> None:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if min_examples_per_client < 1:
        raise ValueError("min_examples_per_client must be at least one")
    required = num_clients * min_examples_per_client
    if num_examples < required:
        raise ValueError(
            f"{num_examples} examples cannot provide {min_examples_per_client} "
            f"examples to each of {num_clients} clients"
        )


def _rebalance_minimum(
    partitions: list[list[int]],
    min_examples_per_client: int,
    rng: np.random.Generator,
) -> None:
    while min(map(len, partitions)) < min_examples_per_client:
        receiver = min(range(len(partitions)), key=lambda index: len(partitions[index]))
        donor = max(range(len(partitions)), key=lambda index: len(partitions[index]))
        if len(partitions[donor]) <= min_examples_per_client:
            raise RuntimeError("unable to satisfy minimum client size")
        position = int(rng.integers(0, len(partitions[donor])))
        partitions[receiver].append(partitions[donor].pop(position))


def _iid_partition(
    num_examples: int,
    num_clients: int,
    *,
    size_imbalance: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    indices = rng.permutation(num_examples)
    if size_imbalance < 0:
        raise ValueError("size_imbalance must be non-negative")
    if size_imbalance == 0:
        chunks = np.array_split(indices, num_clients)
    else:
        weights = rng.lognormal(mean=0.0, sigma=size_imbalance, size=num_clients)
        probabilities = weights / weights.sum()
        counts = rng.multinomial(num_examples, probabilities)
        boundaries = np.cumsum(counts)[:-1]
        chunks = np.split(indices, boundaries)
    return [chunk.astype(int).tolist() for chunk in chunks]


def _dirichlet_partition(
    labels: Sequence[str],
    num_clients: int,
    *,
    alpha: float,
    size_imbalance: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError("dirichlet alpha must be positive")
    if len(labels) == 0:
        raise ValueError("labels cannot be empty")
    client_scale = (
        np.ones(num_clients)
        if size_imbalance == 0
        else rng.lognormal(mean=0.0, sigma=size_imbalance, size=num_clients)
    )
    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    label_array = np.asarray([str(label) for label in labels], dtype=object)
    for label in sorted(set(label_array.tolist())):
        label_indices = np.flatnonzero(label_array == label)
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha)) * client_scale
        probabilities = proportions / proportions.sum()
        counts = rng.multinomial(len(label_indices), probabilities)
        start = 0
        for client_index, count in enumerate(counts):
            end = start + int(count)
            partitions[client_index].extend(label_indices[start:end].astype(int).tolist())
            start = end
    return partitions


def partition_indices(
    *,
    num_examples: int,
    num_clients: int,
    strategy: str,
    seed: int,
    labels: Sequence[str] | None = None,
    dirichlet_alpha: float = 0.5,
    min_examples_per_client: int = 1,
    size_imbalance: float = 0.0,
) -> tuple[tuple[int, ...], ...]:
    """Partition task-level indices without duplicating or dropping examples."""

    _validate_partition_request(num_examples, num_clients, min_examples_per_client)
    rng = np.random.default_rng(seed)
    if strategy == "iid":
        partitions = _iid_partition(
            num_examples,
            num_clients,
            size_imbalance=size_imbalance,
            rng=rng,
        )
    elif strategy == "dirichlet":
        if labels is None or len(labels) != num_examples:
            raise ValueError("dirichlet partitioning requires one label per example")
        partitions = _dirichlet_partition(
            labels,
            num_clients,
            alpha=dirichlet_alpha,
            size_imbalance=size_imbalance,
            rng=rng,
        )
    else:
        raise ValueError(f"unknown partition strategy: {strategy!r}")

    _rebalance_minimum(partitions, min_examples_per_client, rng)
    result = tuple(tuple(sorted(indices)) for indices in partitions)
    flattened = [index for client_indices in result for index in client_indices]
    if sorted(flattened) != list(range(num_examples)):
        raise RuntimeError("partitioning duplicated or dropped examples")
    return result


def split_client_indices(
    indices: Sequence[int],
    ratios: SplitRatios,
    *,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(indices) < 4:
        raise ValueError("each client needs at least four examples")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)

    raw_counts = np.asarray(
        [ratios.support, ratios.query, ratios.validation, ratios.test],
        dtype=np.float64,
    ) * len(shuffled)
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > len(shuffled):
        candidate = int(np.argmax(counts))
        if counts[candidate] <= 1:
            raise ValueError("client split is too small for the requested ratios")
        counts[candidate] -= 1
    while counts.sum() < len(shuffled):
        remainder = raw_counts - counts
        counts[int(np.argmax(remainder))] += 1

    support_end = int(counts[0])
    query_end = support_end + int(counts[1])
    validation_end = query_end + int(counts[2])
    return (
        tuple(sorted(shuffled[:support_end].tolist())),
        tuple(sorted(shuffled[support_end:query_end].tolist())),
        tuple(sorted(shuffled[query_end:validation_end].tolist())),
        tuple(sorted(shuffled[validation_end:].tolist())),
    )


def build_client_partitions(
    *,
    task_id: str,
    task_indices_by_client: Sequence[Sequence[int]],
    ratios: SplitRatios,
    seed: int,
) -> tuple[ClientPartition, ...]:
    partitions: list[ClientPartition] = []
    for client_number, task_indices in enumerate(task_indices_by_client):
        support, query, validation, test = split_client_indices(
            task_indices,
            ratios,
            seed=seed + client_number,
        )
        partitions.append(
            ClientPartition(
                client_id=f"{task_id}/client_{client_number:03d}",
                task_id=task_id,
                support_indices=support,
                query_indices=query,
                validation_indices=validation,
                test_indices=test,
            )
        )
    return tuple(partitions)

