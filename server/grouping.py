"""Cosine-similarity grouping over frozen task-description embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GroupingConfig:
    similarity: str = "cosine"
    strategy: str = "agglomerative"
    assignment_threshold: float = 0.90
    centroid_eps: float = 1.0e-12
    create_new_group: bool = True
    freeze_during_normal_training: bool = True
    target_num_groups: int | None = None

    def __post_init__(self) -> None:
        if self.similarity != "cosine":
            raise ValueError("the revised method supports only cosine similarity")
        if self.strategy not in {"agglomerative", "incremental"}:
            raise ValueError("grouping strategy must be 'agglomerative' or 'incremental'")
        if not -1.0 <= self.assignment_threshold <= 1.0:
            raise ValueError("assignment_threshold must be in [-1, 1]")
        if self.centroid_eps <= 0:
            raise ValueError("centroid_eps must be positive")
        if self.target_num_groups is not None and self.target_num_groups <= 0:
            raise ValueError("target_num_groups must be positive or None")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GroupingConfig:
        removed = {"field_weights", "support_threshold"}.intersection(value)
        if removed:
            raise ValueError(
                "discrete field-support grouping was removed; delete config keys "
                f"{sorted(removed)}"
            )
        raw_target = value.get("target_num_groups", 4)
        return cls(
            similarity=str(value.get("similarity", "cosine")),
            strategy=str(value.get("strategy", "agglomerative")),
            assignment_threshold=float(value.get("assignment_threshold", 0.90)),
            centroid_eps=float(value.get("centroid_eps", 1.0e-12)),
            create_new_group=bool(value.get("create_new_group", True)),
            freeze_during_normal_training=bool(
                value.get("freeze_during_normal_training", True)
            ),
            target_num_groups=None if raw_target is None else int(raw_target),
        )


@dataclass(frozen=True, slots=True)
class AssignmentDecision:
    client_id: str
    group_id: str
    similarity: float
    created: bool
    scores: Mapping[str, float]


@dataclass(slots=True)
class SemanticCluster:
    group_id: str
    members: dict[str, Tensor]
    centroid: Tensor

    @classmethod
    def singleton(cls, group_id: str, client_id: str, embedding: Tensor) -> SemanticCluster:
        return cls(
            group_id=group_id,
            members={client_id: embedding},
            centroid=embedding.clone(),
        )

    def recompute_centroid(self, *, eps: float) -> None:
        if not self.members:
            raise ValueError(f"cannot compute centroid of empty group {self.group_id}")
        mean = torch.stack(tuple(self.members.values()), dim=0).mean(dim=0)
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(norm) or float(norm) <= eps:
            raise ValueError(f"group {self.group_id} has an undefined semantic centroid")
        self.centroid = mean / norm


def normalize_embedding(embedding: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    """Detach one semantic vector, move it to CPU FP32, and L2-normalize it."""

    if embedding.ndim != 1:
        raise ValueError(f"task embedding must be a vector, got {embedding.shape}")
    if not embedding.is_floating_point():
        raise TypeError("task embedding must be floating point")
    vector = embedding.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(vector).all():
        raise ValueError("task embedding contains a non-finite value")
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= eps:
        raise ValueError("task embedding must have non-zero norm")
    return vector / norm


class SemanticGrouper:
    """Incrementally implement Eqs. (8)--(11) with deterministic tie-breaking.

    Initial fitting processes client IDs in sorted order. Once frozen, existing
    memberships remain unchanged, while a previously unseen client can still be
    routed to an existing group or used to create a new group.
    """

    def __init__(self, config: GroupingConfig) -> None:
        self.config = config
        self._clusters: dict[str, SemanticCluster] = {}
        self._assignments: dict[str, str] = {}
        self._next_group_index = 0
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def assignments(self) -> dict[str, str]:
        return dict(self._assignments)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._clusters))

    def members(self, group_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._clusters[group_id].members))

    def centroid(self, group_id: str) -> Tensor:
        return self._clusters[group_id].centroid.clone()

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    def _new_group_id(self) -> str:
        while True:
            group_id = f"group_{self._next_group_index:04d}"
            self._next_group_index += 1
            if group_id not in self._clusters:
                return group_id

    def _scores(self, embedding: Tensor) -> dict[str, float]:
        return {
            group_id: float(torch.dot(embedding, cluster.centroid))
            for group_id, cluster in sorted(self._clusters.items())
        }

    def _remove_existing_member(self, client_id: str) -> None:
        previous_group = self._assignments.pop(client_id)
        cluster = self._clusters[previous_group]
        del cluster.members[client_id]
        if cluster.members:
            cluster.recompute_centroid(eps=self.config.centroid_eps)
        else:
            del self._clusters[previous_group]

    def assign(self, client_id: str, embedding: Tensor) -> AssignmentDecision:
        client_id = str(client_id).strip()
        if not client_id:
            raise ValueError("client_id must be non-empty")
        vector = normalize_embedding(embedding, eps=self.config.centroid_eps)

        if client_id in self._assignments and self._frozen:
            group_id = self._assignments[client_id]
            scores = self._scores(vector)
            return AssignmentDecision(
                client_id=client_id,
                group_id=group_id,
                similarity=scores[group_id],
                created=False,
                scores=scores,
            )
        if client_id in self._assignments:
            self._remove_existing_member(client_id)

        scores = self._scores(vector)
        # ``max`` keeps the first maximum, so sorting implements the paper's
        # smallest-index tie break without perturbing cosine scores.
        best_group = max(sorted(scores), key=scores.__getitem__) if scores else None
        best_score = scores[best_group] if best_group is not None else float("-inf")
        should_create = best_group is None or best_score < self.config.assignment_threshold

        if should_create:
            if not self.config.create_new_group:
                if best_group is None:
                    raise RuntimeError("cannot assign a client when no group exists")
                group_id = best_group
                created = False
            else:
                group_id = self._new_group_id()
                self._clusters[group_id] = SemanticCluster.singleton(
                    group_id, client_id, vector
                )
                self._assignments[client_id] = group_id
                return AssignmentDecision(
                    client_id=client_id,
                    group_id=group_id,
                    similarity=1.0,
                    created=True,
                    scores=scores,
                )
        else:
            group_id = best_group
            created = False

        cluster = self._clusters[group_id]
        cluster.members[client_id] = vector
        cluster.recompute_centroid(eps=self.config.centroid_eps)
        self._assignments[client_id] = group_id
        return AssignmentDecision(
            client_id=client_id,
            group_id=group_id,
            similarity=best_score,
            created=created,
            scores=scores,
        )

    def fit(self, embeddings: Mapping[str, Tensor]) -> dict[str, str]:
        if not embeddings:
            raise ValueError("embeddings cannot be empty")
        self._clusters.clear()
        self._assignments.clear()
        self._next_group_index = 0
        self._frozen = False
        if self.config.strategy == "agglomerative":
            self._fit_agglomerative(embeddings)
        else:
            for client_id in sorted(embeddings):
                self.assign(client_id, embeddings[client_id])
        if self.config.freeze_during_normal_training:
            self.freeze()
        return self.assignments

    def _fit_agglomerative(self, embeddings: Mapping[str, Tensor]) -> None:
        """Deterministic centroid-linkage clustering independent of input order."""

        normalized = {
            item_id: normalize_embedding(
                embedding,
                eps=self.config.centroid_eps,
            )
            for item_id, embedding in sorted(embeddings.items())
        }
        clusters: list[tuple[str, ...]] = [(item_id,) for item_id in normalized]

        def centroid(members: tuple[str, ...]) -> Tensor:
            mean = torch.stack([normalized[item_id] for item_id in members]).mean(dim=0)
            return normalize_embedding(mean, eps=self.config.centroid_eps)

        while len(clusters) > 1:
            if (
                self.config.target_num_groups is not None
                and len(clusters) <= self.config.target_num_groups
            ):
                break
            best: tuple[float, tuple[str, ...], tuple[str, ...], int, int] | None = None
            for left_index, left in enumerate(clusters):
                left_centroid = centroid(left)
                for right_index in range(left_index + 1, len(clusters)):
                    right = clusters[right_index]
                    score = float(torch.dot(left_centroid, centroid(right)))
                    candidate = (score, left, right, left_index, right_index)
                    if best is None or score > best[0] or (
                        score == best[0] and (left, right) < (best[1], best[2])
                    ):
                        best = candidate
            if best is None or best[0] < self.config.assignment_threshold:
                break
            _, left, right, left_index, right_index = best
            merged = tuple(sorted((*left, *right)))
            clusters = [
                members
                for index, members in enumerate(clusters)
                if index not in {left_index, right_index}
            ]
            clusters.append(merged)
            clusters.sort()

        for members in sorted(clusters):
            group_id = self._new_group_id()
            member_embeddings = {item_id: normalized[item_id] for item_id in members}
            self._clusters[group_id] = SemanticCluster(
                group_id=group_id,
                members=member_embeddings,
                centroid=centroid(members),
            )
            for item_id in members:
                self._assignments[item_id] = group_id
