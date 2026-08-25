"""Joint server update for the single coordinate generator shared by all groups."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from client.state import ClientRoundResult
from model.coordinate_generator import CoordinateGenerator
from server.group_state import GroupState


@dataclass(frozen=True, slots=True)
class ServerOptimizerConfig:
    name: str = "adamw"
    lr: float = 1.0e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 1.0
    coordinate_regularization: float = 1.0e-4
    group_weighting: str = "uniform"
    client_weighting: str = "query_examples"

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError("the prototype currently supports only AdamW")
        if self.lr < 0 or self.weight_decay < 0 or self.coordinate_regularization < 0:
            raise ValueError("optimizer coefficients must be non-negative")
        if len(self.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.group_weighting not in {"uniform", "query_examples"}:
            raise ValueError("group_weighting must be 'uniform' or 'query_examples'")
        if self.client_weighting not in {"uniform", "query_examples"}:
            raise ValueError("client_weighting must be 'uniform' or 'query_examples'")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ServerOptimizerConfig:
        raw_betas = tuple(float(beta) for beta in value.get("betas", (0.9, 0.999)))
        if len(raw_betas) != 2:
            raise ValueError("betas must contain exactly two values")
        return cls(
            name=str(value.get("name", "adamw")).lower(),
            lr=float(value.get("lr", 1.0e-3)),
            betas=(raw_betas[0], raw_betas[1]),
            weight_decay=float(value.get("weight_decay", 1.0e-4)),
            max_grad_norm=float(value.get("max_grad_norm", 1.0)),
            coordinate_regularization=float(
                value.get("coordinate_regularization", 1.0e-4)
            ),
            group_weighting=str(value.get("group_weighting", "uniform")),
            client_weighting=str(value.get("client_weighting", "query_examples")),
        )


@dataclass(frozen=True, slots=True)
class AggregationSummary:
    active_groups: tuple[str, ...]
    clients: int
    query_examples: int
    mean_query_loss: float
    mean_support_loss: float
    gradient_norm: float
    gradient_surrogate: float


def _normalized_weights(results: Sequence[ClientRoundResult], mode: str) -> Tensor:
    if mode == "uniform":
        values = torch.ones(len(results), dtype=torch.float64)
    else:
        values = torch.tensor(
            [result.query_examples for result in results],
            dtype=torch.float64,
        )
    return values / values.sum()


class SharedGeneratorTrainer(nn.Module):
    """Own exactly one generator and apply the cross-group update in Eq. (35)."""

    def __init__(
        self,
        generator: CoordinateGenerator,
        config: ServerOptimizerConfig,
    ) -> None:
        super().__init__()
        if generator.settings.dropout != 0.0:
            raise ValueError(
                "server VJP recomputation must be deterministic; set generator dropout to 0"
            )
        self.generator = generator
        self.settings = config
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

    def initial_coordinates(self, task_embedding: Tensor, group: GroupState) -> Tensor:
        return group.initial_coordinates(self.generator, task_embedding)

    def _group_weights(
        self,
        grouped: Mapping[str, Sequence[ClientRoundResult]],
    ) -> dict[str, float]:
        group_ids = sorted(grouped)
        if self.settings.group_weighting == "uniform":
            return {group_id: 1.0 / len(group_ids) for group_id in group_ids}
        totals = {
            group_id: sum(result.query_examples for result in results)
            for group_id, results in grouped.items()
        }
        denominator = float(sum(totals.values()))
        return {group_id: totals[group_id] / denominator for group_id in group_ids}

    def update(
        self,
        *,
        results: Sequence[ClientRoundResult],
        task_embeddings: Mapping[str, Tensor],
        groups: Mapping[str, GroupState],
    ) -> AggregationSummary:
        if not results:
            raise ValueError("cannot update the shared generator without client results")

        grouped: dict[str, list[ClientRoundResult]] = defaultdict(list)
        for result in results:
            if result.group_id not in groups:
                raise KeyError(f"unknown group {result.group_id!r}")
            if result.client_id not in task_embeddings:
                raise KeyError(f"missing task embedding for client {result.client_id!r}")
            grouped[result.group_id].append(result)

        group_weights = self._group_weights(grouped)
        parameter = next(self.generator.parameters())
        surrogate = torch.zeros((), device=parameter.device, dtype=parameter.dtype)

        weighted_query_loss = 0.0
        weighted_support_loss = 0.0
        for group_id in sorted(grouped):
            group_results = grouped[group_id]
            client_weights = _normalized_weights(
                group_results,
                self.settings.client_weighting,
            )
            group = groups[group_id]
            task_batch = torch.stack(
                [
                    task_embeddings[result.client_id].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    for result in group_results
                ],
                dim=0,
            )
            prototype = group.semantic_prototype.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).expand_as(task_batch)
            coordinates = self.generator(task_batch, prototype)
            feedback = torch.stack(
                [
                    result.coordinate_feedback.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    for result in group_results
                ],
                dim=0,
            ).detach()
            expected_initial = torch.stack(
                [
                    result.initial_coordinates.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    for result in group_results
                ],
                dim=0,
            )
            if not torch.allclose(
                coordinates.detach(),
                expected_initial,
                atol=1.0e-5,
                rtol=1.0e-4,
            ):
                raise RuntimeError(
                    "recomputed coordinates differ from those used by clients; "
                    "do not update the global generator before round aggregation"
                )

            client_terms = (coordinates * feedback).sum(dim=-1)
            client_terms = client_terms + 0.5 * self.settings.coordinate_regularization * (
                coordinates.square().sum(dim=-1)
            )
            weights = client_weights.to(device=parameter.device, dtype=parameter.dtype)
            group_weight = group_weights[group_id]
            surrogate = surrogate + group_weight * torch.dot(weights, client_terms)
            weighted_query_loss += group_weight * sum(
                float(weight) * result.query_loss
                for weight, result in zip(client_weights, group_results, strict=True)
            )
            weighted_support_loss += group_weight * sum(
                float(weight) * result.mean_support_loss
                for weight, result in zip(client_weights, group_results, strict=True)
            )

        self.optimizer.zero_grad(set_to_none=True)
        surrogate.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.settings.max_grad_norm,
        )
        self.optimizer.step()

        return AggregationSummary(
            active_groups=tuple(sorted(grouped)),
            clients=len(results),
            query_examples=sum(result.query_examples for result in results),
            mean_query_loss=weighted_query_loss,
            mean_support_loss=weighted_support_loss,
            gradient_norm=float(gradient_norm),
            gradient_surrogate=float(surrogate.detach()),
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "module": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(state["module"])
        self.optimizer.load_state_dict(state["optimizer"])

