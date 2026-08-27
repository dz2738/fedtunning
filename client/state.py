"""Persistent client metadata and round-level result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from torch import Tensor, norm

from data.schema import TaskDescription


class MetaGradientMode(StrEnum):
    FIRST_ORDER = "first_order"
    COORDINATE_SECOND_ORDER = "coordinate_second_order"
    FULL_SECOND_ORDER = "full_second_order"


@dataclass(frozen=True, slots=True)
class InnerLoopConfig:
    steps: int = 3
    support_batch_size: int = 4
    query_batch_size: int = 8
    coordinate_lr: float = 0.1
    residual_lr: float = 0.01
    coordinate_weight_decay: float = 1.0e-4
    residual_weight_decay: float = 1.0e-4
    residual_max_norm: float | None = None
    residual_update_max_norm: float | None = None
    project_residual: bool = True
    reset_residual_each_round: bool = True
    meta_gradient: MetaGradientMode = (
        MetaGradientMode.COORDINATE_SECOND_ORDER
    )


    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.support_batch_size <= 0 or self.query_batch_size <= 0:
            raise ValueError("support_batch_size and query_batch_size must be positive")
        if self.coordinate_lr < 0 or self.residual_lr < 0:
            raise ValueError("inner-loop learning rates must be non-negative")
        if self.coordinate_weight_decay < 0 or self.residual_weight_decay < 0:
            raise ValueError("inner-loop regularization coefficients must be non-negative")
        if self.residual_max_norm is not None and self.residual_max_norm <= 0:
            raise ValueError("residual_max_norm must be positive")
        if self.residual_update_max_norm is not None and self.residual_update_max_norm <= 0:
            raise ValueError("residual_update_max_norm must be positive")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> InnerLoopConfig:
        return cls(
            steps=int(value.get("steps", 5)),
            support_batch_size=int(
                value.get("support_batch_size", 4)
            ),
            query_batch_size=int(
                value.get("query_batch_size", 8)
            ),
            coordinate_lr=float(
                value.get("coordinate_lr", 0.1)
            ),
            residual_lr=float(
                value.get("residual_lr", 0.01)
            ),
            coordinate_weight_decay=float(
                value.get(
                    "coordinate_weight_decay",
                    1.0e-4,
                )
            ),
            residual_weight_decay=float(
                value.get(
                    "residual_weight_decay",
                    1.0e-4,
                )
            ),
            residual_max_norm=(
                0.01
                if value.get("residual_max_norm") is None
                else float(value["residual_max_norm"])
            ),
            residual_update_max_norm=(
                0.05
                if value.get("residual_update_max_norm") is None
                else float(
                    value["residual_update_max_norm"]
                )
            ),
            project_residual=bool(
                value.get("project_residual", True)
            ),
            reset_residual_each_round=bool(
                value.get(
                    "reset_residual_each_round",
                    True,
                )
            ),
            meta_gradient=MetaGradientMode(
                str(
                    value.get(
                        "meta_gradient",
                        MetaGradientMode.COORDINATE_SECOND_ORDER,
                    )
                )
            ),
        )


@dataclass(slots=True)
class ClientState:
    client_id: str
    task_id: str
    description: TaskDescription
    group_id: str | None = None
    deployment_residual: Tensor | None = None
    metric_history: list[dict[str, float]] = field(default_factory=list)

    def assign_group(self, group_id: str) -> None:
        group_id = str(group_id).strip()
        if not group_id:
            raise ValueError("group_id must be non-empty")
        self.group_id = group_id

    def record_metrics(self, **metrics: float) -> None:
        self.metric_history.append({name: float(value) for name, value in metrics.items()})


@dataclass(frozen=True, slots=True)
class ClientRoundResult:
    client_id: str
    group_id: str
    meta_gradient: MetaGradientMode
    coordinate_feedback: Tensor
    initial_coordinates: Tensor
    terminal_coordinates: Tensor
    terminal_residual: Tensor | None
    residual_energy: float
    mean_support_loss: float
    query_loss: float
    support_examples: int
    query_examples: int

    def __post_init__(self) -> None:
        if self.coordinate_feedback.ndim != 1:
            raise ValueError("coordinate_feedback must be a vector")
        if self.initial_coordinates.shape != self.coordinate_feedback.shape:
            raise ValueError("initial coordinates and feedback must have equal shapes")
        if self.terminal_coordinates.shape != self.coordinate_feedback.shape:
            raise ValueError("terminal coordinates and feedback must have equal shapes")
        if self.residual_energy < 0:
            raise ValueError("residual_energy must be non-negative")
        if self.support_examples <= 0 or self.query_examples <= 0:
            raise ValueError("client round must use support and query examples")
        

@dataclass(frozen=True, slots=True)
class ClientEvaluationResult:
    """Result returned by one client during evaluation."""

    client_id: str
    group_id: str
    adaptation_steps: int
    mean_support_loss: float | None
    test_loss: float
    residual_energy: float
    support_examples: int
    test_examples: int

    def __post_init__(self) -> None:
        if self.adaptation_steps < 0:
            raise ValueError(
                "adaptation_steps must be non-negative"
            )
        if self.residual_energy < 0:
            raise ValueError(
                "residual_energy must be non-negative"
            )
        if self.support_examples <= 0 or self.test_examples <= 0:
            raise ValueError(
                "evaluation requires support and test examples"
            )

