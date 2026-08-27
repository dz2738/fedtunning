"""Persistent client metadata and round-level result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from torch import Tensor

from data.schema import DataSplit, TaskDescription


class MetaGradientMode(StrEnum):
    FIRST_ORDER = "first_order"
    COORDINATE_SECOND_ORDER = "coordinate_second_order"
    FULL_SECOND_ORDER = "full_second_order"


@dataclass(frozen=True, slots=True)
class InnerLoopConfig:
    steps: int = 5
    support_batch_size: int = 4
    query_batch_size: int = 8
    coordinate_lr: float = 0.001
    residual_lr: float = 0.002
    coordinate_weight_decay: float = 1.0e-4
    residual_weight_decay: float = 1.0e-4
    project_residual: bool = True
    reset_residual_each_round: bool = True
    meta_gradient: MetaGradientMode = MetaGradientMode.COORDINATE_SECOND_ORDER
    second_order_steps: int | None = None
    hessian_damping: float = 0.0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.support_batch_size <= 0 or self.query_batch_size <= 0:
            raise ValueError("support_batch_size and query_batch_size must be positive")
        if self.coordinate_lr < 0 or self.residual_lr < 0:
            raise ValueError("inner-loop learning rates must be non-negative")
        if self.coordinate_weight_decay < 0 or self.residual_weight_decay < 0:
            raise ValueError("inner-loop regularization coefficients must be non-negative")
        if self.second_order_steps is not None and self.second_order_steps <= 0:
            raise ValueError("second_order_steps must be positive or None")
        if self.hessian_damping < 0:
            raise ValueError("hessian_damping must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InnerLoopConfig:
        return cls(
            steps=int(value.get("steps", 5)),
            support_batch_size=int(value.get("support_batch_size", 4)),
            query_batch_size=int(value.get("query_batch_size", 8)),
            coordinate_lr=float(value.get("coordinate_lr", 0.001)),
            residual_lr=float(value.get("residual_lr", 0.002)),
            coordinate_weight_decay=float(value.get("coordinate_weight_decay", 1.0e-4)),
            residual_weight_decay=float(value.get("residual_weight_decay", 1.0e-4)),
            project_residual=bool(value.get("project_residual", True)),
            reset_residual_each_round=bool(value.get("reset_residual_each_round", True)),
            meta_gradient=MetaGradientMode(
                str(value.get("meta_gradient", MetaGradientMode.COORDINATE_SECOND_ORDER))
            ),
            second_order_steps=(
                None
                if value.get("second_order_steps") is None
                else int(value["second_order_steps"])
            ),
            hessian_damping=float(value.get("hessian_damping", 0.0)),
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
class MetaGradientStepDiagnostic:
    step: int
    vector_norm_before: float
    hessian_vector_norm: float
    vector_norm_after: float
    rayleigh_quotient: float


@dataclass(frozen=True, slots=True)
class ClientRoundResult:
    client_id: str
    group_id: str
    meta_gradient: MetaGradientMode
    coordinate_feedback: Tensor
    initial_coordinates: Tensor
    terminal_coordinates: Tensor
    # Client-clipped residual uploaded only on scheduled maintenance rounds.
    # ``residual_energy`` below is computed from the unclipped local residual.
    terminal_residual: Tensor | None
    residual_energy: float
    mean_support_loss: float
    query_loss: float
    support_examples: int
    query_examples: int
    prompt_energy: float | None = None
    second_order_diagnostics: tuple[MetaGradientStepDiagnostic, ...] = ()
    support_losses: tuple[float, ...] = ()
    support_monitor_losses: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.coordinate_feedback.ndim != 1:
            raise ValueError("coordinate_feedback must be a vector")
        if self.initial_coordinates.shape != self.coordinate_feedback.shape:
            raise ValueError("initial coordinates and feedback must have equal shapes")
        if self.terminal_coordinates.shape != self.coordinate_feedback.shape:
            raise ValueError("terminal coordinates and feedback must have equal shapes")
        if self.residual_energy < 0:
            raise ValueError("residual_energy must be non-negative")
        if self.prompt_energy is not None and self.prompt_energy <= 0:
            raise ValueError("prompt_energy must be positive when provided")
        if self.support_examples <= 0 or self.query_examples <= 0:
            raise ValueError("client round must use support and query examples")


@dataclass(frozen=True, slots=True)
class ClientEvaluationResult:
    client_id: str
    group_id: str
    adaptation_steps: int
    mean_support_loss: float | None
    test_loss: float
    residual_energy: float
    support_examples: int
    test_examples: int
    split: DataSplit = DataSplit.TEST
    metric_name: str | None = None
    metric_value: float | None = None

    def __post_init__(self) -> None:
        if self.adaptation_steps < 0:
            raise ValueError("adaptation_steps must be non-negative")
        if self.residual_energy < 0:
            raise ValueError("residual_energy must be non-negative")
        if self.support_examples <= 0 or self.test_examples <= 0:
            raise ValueError("evaluation requires support and held-out examples")
        if (self.metric_name is None) != (self.metric_value is None):
            raise ValueError("metric_name and metric_value must be provided together")
