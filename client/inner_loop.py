"""Local support adaptation and meta-gradient feedback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from client.state import (
    InnerLoopConfig,
    MetaGradientMode,
    MetaGradientStepDiagnostic,
)
from model.prompt_subspace import PromptSubspace

SupportLoss = Callable[[Tensor, int], Tensor]
QueryLoss = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    coordinate_feedback: Tensor
    initial_coordinates: Tensor
    terminal_coordinates: Tensor
    terminal_residual: Tensor
    support_losses: tuple[float, ...]
    support_monitor_losses: tuple[float, ...]
    query_loss: float
    second_order_diagnostics: tuple[MetaGradientStepDiagnostic, ...] = ()

    @property
    def mean_support_loss(self) -> float:
        return sum(self.support_losses) / len(self.support_losses)


def _scalar_loss(name: str, loss: Tensor) -> Tensor:
    if loss.ndim != 0:
        raise ValueError(f"{name} must return a scalar tensor, got shape {loss.shape}")
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{name} returned a non-finite value")
    return loss


def _support_objective(
    task_loss: Tensor,
    coordinates: Tensor,
    residual: Tensor,
    config: InnerLoopConfig,
) -> Tensor:
    coordinate_penalty = 0.5 * config.coordinate_weight_decay * coordinates.square().sum()
    residual_penalty = 0.5 * config.residual_weight_decay * residual.square().sum()
    return task_loss + coordinate_penalty + residual_penalty


def _project_if_requested(
    residual: Tensor,
    subspace: PromptSubspace,
    config: InnerLoopConfig,
) -> Tensor:
    return subspace.project_residual(residual) if config.project_residual else residual


def _monitor_support_loss(
    monitor_loss: QueryLoss | None,
    subspace: PromptSubspace,
    coordinates: Tensor,
    residual: Tensor,
) -> float | None:
    if monitor_loss is None:
        return None
    with torch.no_grad():
        prompt = subspace(coordinates.detach(), residual.detach())
        loss = _scalar_loss("support_monitor_loss", monitor_loss(prompt))
    return float(loss)


def _detached_adaptation(
    support_loss: SupportLoss,
    support_monitor_loss: QueryLoss | None,
    subspace: PromptSubspace,
    initial_coordinates: Tensor,
    initial_residual: Tensor | None,
    config: InnerLoopConfig,
) -> tuple[
    list[Tensor],
    list[Tensor],
    tuple[float, ...],
    tuple[float, ...],
]:
    coordinates = (
        initial_coordinates
        .detach()
        .requires_grad_(True)
    )

    residual = (
        torch.zeros_like(subspace.center)
        if initial_residual is None
        else initial_residual.to(
            device=subspace.center.device,
            dtype=subspace.center.dtype,
        )
    )

    residual = _project_if_requested(
        residual,
        subspace,
        config,
    )

    residual = (
        residual
        .detach()
        .requires_grad_(True)
    )

    coordinate_states = [coordinates.detach()]
    residual_states = [residual.detach()]
    loss_values: list[float] = []
    monitor_values: list[float] = []
    initial_monitor = _monitor_support_loss(
        support_monitor_loss,
        subspace,
        coordinates,
        residual,
    )
    if initial_monitor is not None:
        monitor_values.append(initial_monitor)

    for step in range(config.steps):
        prompt = subspace(
            coordinates,
            residual,
        )

        task_loss = _scalar_loss(
            "support_loss",
            support_loss(prompt, step),
        )

        objective = _support_objective(
            task_loss,
            coordinates,
            residual,
            config,
        )

        coordinate_grad, residual_grad = (
            torch.autograd.grad(
                objective,
                (coordinates, residual),
                create_graph=False,
            )
        )

        coordinates = (
            coordinates
            - config.coordinate_lr * coordinate_grad
        )
        coordinates = (
            coordinates
            .detach()
            .requires_grad_(True)
        )

        residual_update = -config.residual_lr * residual_grad

        next_residual = (
            residual + residual_update
        )

        next_residual = _project_if_requested(
            next_residual,
            subspace,
            config,
        )

        residual = (
            next_residual
            .detach()
            .requires_grad_(True)
        )

        coordinate_states.append(
            coordinates.detach()
        )
        residual_states.append(
            residual.detach()
        )
        loss_values.append(
            float(task_loss.detach())
        )
        monitor_value = _monitor_support_loss(
            support_monitor_loss,
            subspace,
            coordinates,
            residual,
        )
        if monitor_value is not None:
            monitor_values.append(monitor_value)

    return (
        coordinate_states,
        residual_states,
        tuple(loss_values),
        tuple(monitor_values),
    )

def _query_coordinate_gradient(
    query_loss: QueryLoss,
    subspace: PromptSubspace,
    coordinates: Tensor,
    residual: Tensor,
) -> tuple[Tensor, Tensor]:
    coordinates = coordinates.detach().requires_grad_(True)
    prompt = subspace(coordinates, residual.detach())
    loss = _scalar_loss("query_loss", query_loss(prompt))
    gradient = torch.autograd.grad(loss, coordinates)[0]
    return loss, gradient


def _coordinate_second_order_feedback(
    support_loss: SupportLoss,
    query_loss: QueryLoss,
    subspace: PromptSubspace,
    coordinate_states: list[Tensor],
    residual_states: list[Tensor],
    config: InnerLoopConfig,
) -> tuple[Tensor, Tensor, tuple[MetaGradientStepDiagnostic, ...]]:
    query_value, vector = _query_coordinate_gradient(
        query_loss,
        subspace,
        coordinate_states[-1],
        residual_states[-1],
    )
    diagnostics: list[MetaGradientStepDiagnostic] = []
    backward_steps = min(
        config.steps,
        config.second_order_steps or config.steps,
    )
    first_step = config.steps - backward_steps
    for step in range(config.steps - 1, first_step - 1, -1):
        coordinates = coordinate_states[step].detach().requires_grad_(True)
        residual = residual_states[step].detach()
        prompt = subspace(coordinates, residual)
        task_loss = _scalar_loss("support_loss", support_loss(prompt, step))
        objective = _support_objective(task_loss, coordinates, residual, config)
        coordinate_grad = torch.autograd.grad(
            objective,
            coordinates,
            create_graph=True,
        )[0]
        if coordinate_grad.requires_grad:
            hessian_vector = torch.autograd.grad(
                coordinate_grad,
                coordinates,
                grad_outputs=vector,
            )[0]
        else:
            hessian_vector = torch.zeros_like(coordinates)
        vector_before = vector
        damped_hessian_vector = (
            hessian_vector + config.hessian_damping * vector_before
        )
        vector = vector - config.coordinate_lr * damped_hessian_vector
        denominator = vector_before.detach().float().square().sum()
        rayleigh = (
            torch.dot(
                vector_before.detach().float(),
                hessian_vector.detach().float(),
            )
            / denominator.clamp_min(1.0e-30)
        )
        diagnostics.append(
            MetaGradientStepDiagnostic(
                step=step,
                vector_norm_before=float(
                    torch.linalg.vector_norm(vector_before.detach().float())
                ),
                hessian_vector_norm=float(
                    torch.linalg.vector_norm(hessian_vector.detach().float())
                ),
                vector_norm_after=float(
                    torch.linalg.vector_norm(vector.detach().float())
                ),
                rayleigh_quotient=float(rayleigh),
            )
        )
    return query_value, vector, tuple(diagnostics)


def _full_second_order_adaptation(
    support_loss: SupportLoss,
    support_monitor_loss: QueryLoss | None,
    query_loss: QueryLoss,
    subspace: PromptSubspace,
    initial_coordinates: Tensor,
    initial_residual: Tensor | None,
    config: InnerLoopConfig,
) -> AdaptationResult:
    coordinates_0 = initial_coordinates
    if not coordinates_0.requires_grad:
        coordinates_0 = coordinates_0.detach().requires_grad_(True)
    coordinates = coordinates_0
    residual = (
        torch.zeros_like(subspace.center)
        if initial_residual is None
        else initial_residual.to(
            device=subspace.center.device,
            dtype=subspace.center.dtype,
        )
    )
    residual = _project_if_requested(residual, subspace, config).detach().requires_grad_(True)
    loss_values: list[float] = []
    monitor_values: list[float] = []
    initial_monitor = _monitor_support_loss(
        support_monitor_loss,
        subspace,
        coordinates,
        residual,
    )
    if initial_monitor is not None:
        monitor_values.append(initial_monitor)

    for step in range(config.steps):
        prompt = subspace(coordinates, residual)
        task_loss = _scalar_loss("support_loss", support_loss(prompt, step))
        objective = _support_objective(task_loss, coordinates, residual, config)
        coordinate_grad, residual_grad = torch.autograd.grad(
            objective,
            (coordinates, residual),
            create_graph=True,
        )
        coordinates = coordinates - config.coordinate_lr * coordinate_grad
        residual = _project_if_requested(
            residual - config.residual_lr * residual_grad,
            subspace,
            config,
        )
        loss_values.append(float(task_loss.detach()))
        monitor_value = _monitor_support_loss(
            support_monitor_loss,
            subspace,
            coordinates,
            residual,
        )
        if monitor_value is not None:
            monitor_values.append(monitor_value)

    query_value = _scalar_loss("query_loss", query_loss(subspace(coordinates, residual)))
    feedback = torch.autograd.grad(query_value, coordinates_0)[0]
    return AdaptationResult(
        coordinate_feedback=feedback.detach(),
        initial_coordinates=coordinates_0.detach(),
        terminal_coordinates=coordinates.detach(),
        terminal_residual=residual.detach(),
        support_losses=tuple(loss_values),
        support_monitor_losses=tuple(monitor_values),
        query_loss=float(query_value.detach()),
    )


def adapt_and_compute_feedback(
    *,
    support_loss: SupportLoss,
    support_monitor_loss: QueryLoss | None = None,
    query_loss: QueryLoss,
    subspace: PromptSubspace,
    initial_coordinates: Tensor,
    config: InnerLoopConfig,
    initial_residual: Tensor | None = None,
) -> AdaptationResult:
    """Run the configured local adaptation and return dL_query/dc_initial."""

    if initial_coordinates.ndim != 1:
        raise ValueError("the prototype adapts one client coordinate vector at a time")
    if initial_coordinates.shape[0] != subspace.num_basis:
        raise ValueError(
            f"expected {subspace.num_basis} coordinates, got {initial_coordinates.shape[0]}"
        )
    if initial_coordinates.device != subspace.center.device:
        raise ValueError("initial coordinates and prompt subspace must share a device")

    if config.meta_gradient is MetaGradientMode.FULL_SECOND_ORDER:
        return _full_second_order_adaptation(
            support_loss,
            support_monitor_loss,
            query_loss,
            subspace,
            initial_coordinates,
            initial_residual,
            config,
        )

    coordinate_states, residual_states, support_losses, support_monitor_losses = (
        _detached_adaptation(
        support_loss,
        support_monitor_loss,
        subspace,
        initial_coordinates,
        initial_residual,
        config,
        )
    )
    if config.meta_gradient is MetaGradientMode.FIRST_ORDER:
        query_value, feedback = _query_coordinate_gradient(
            query_loss,
            subspace,
            coordinate_states[-1],
            residual_states[-1],
        )
    elif config.meta_gradient is MetaGradientMode.COORDINATE_SECOND_ORDER:
        query_value, feedback, diagnostics = _coordinate_second_order_feedback(
            support_loss,
            query_loss,
            subspace,
            coordinate_states,
            residual_states,
            config,
        )
    else:
        raise ValueError(f"unsupported meta-gradient mode: {config.meta_gradient}")

    return AdaptationResult(
        coordinate_feedback=feedback.detach(),
        initial_coordinates=initial_coordinates.detach(),
        terminal_coordinates=coordinate_states[-1],
        terminal_residual=residual_states[-1],
        support_losses=support_losses,
        support_monitor_losses=support_monitor_losses,
        query_loss=float(query_value.detach()),
        second_order_diagnostics=(
            diagnostics
            if config.meta_gradient is MetaGradientMode.COORDINATE_SECOND_ORDER
            else ()
        ),
    )