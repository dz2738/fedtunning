"""Empirical transfer matrices and semantic-similarity diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def signed_gain(method: float, baseline: float, *, higher_is_better: bool = True) -> float:
    return float(method - baseline if higher_is_better else baseline - method)


def relative_gain(
    method: float,
    baseline: float,
    *,
    higher_is_better: bool = True,
    eps: float = 1.0e-12,
) -> float:
    gain = signed_gain(method, baseline, higher_is_better=higher_is_better)
    return gain / max(abs(float(baseline)), eps)


@dataclass(frozen=True, slots=True)
class TransferMatrix:
    client_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.client_ids), len(self.client_ids))
        if self.values.shape != expected:
            raise ValueError(f"transfer matrix must have shape {expected}, got {self.values.shape}")
        if len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("client_ids must be unique")

    @classmethod
    def from_nested_mapping(
        cls,
        values: Mapping[str, Mapping[str, float]],
    ) -> TransferMatrix:
        client_ids = tuple(sorted(values))
        if any(set(row) != set(client_ids) for row in values.values()):
            raise ValueError("every transfer-matrix row must contain every client")
        matrix = np.asarray(
            [[values[source][target] for target in client_ids] for source in client_ids],
            dtype=np.float64,
        )
        return cls(client_ids=client_ids, values=matrix)

    def off_diagonal(self) -> np.ndarray:
        mask = ~np.eye(len(self.client_ids), dtype=bool)
        return self.values[mask]


def binary_roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Compute tie-aware ROC-AUC using the Mann-Whitney rank statistic."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=bool)
    if score_array.ndim != 1 or score_array.shape != label_array.shape:
        raise ValueError("scores and labels must be equally sized vectors")
    positive_count = int(label_array.sum())
    negative_count = len(label_array) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")

    order = np.argsort(score_array, kind="mergesort")
    ranks = np.empty(len(score_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score_array[order[end]] == score_array[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[label_array].sum())
    statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return statistic / (positive_count * negative_count)


def semantic_transfer_auc(
    semantic_similarities: Sequence[float],
    empirical_gains: Sequence[float],
    *,
    positive_threshold: float = 0.0,
) -> float:
    if len(semantic_similarities) != len(empirical_gains):
        raise ValueError("semantic similarities and empirical gains must have equal lengths")
    labels = [gain > positive_threshold for gain in empirical_gains]
    return binary_roc_auc(semantic_similarities, labels)
