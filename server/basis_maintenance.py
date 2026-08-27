"""Low-frequency group prompt-center and basis maintenance."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from client.state import ClientRoundResult
from server.group_state import GroupState


@dataclass(frozen=True, slots=True)
class BasisMaintenanceConfig:
    enabled: bool = True
    interval: int = 10
    energy_mode: str = "relative"
    residual_energy_threshold: float = 0.1
    residual_clip_norm: float = 5.0
    center_update_rate: float = 0.1
    explained_energy_ratio: float = 0.9
    max_replaced_basis: int = 1
    usage_window: int = 20
    orthogonality_tolerance: float = 1.0e-5
    numerical_reorthogonalize: bool = True

    def __post_init__(self) -> None:
        if self.interval <= 0 or self.usage_window <= 0:
            raise ValueError("maintenance interval and usage_window must be positive")
        if self.residual_energy_threshold < 0 or self.residual_clip_norm <= 0:
            raise ValueError("residual energy must be non-negative and clip norm positive")
        if self.energy_mode not in {"absolute", "relative"}:
            raise ValueError("energy_mode must be 'absolute' or 'relative'")
        if not 0.0 <= self.center_update_rate <= 1.0:
            raise ValueError("center_update_rate must be in [0, 1]")
        if not 0.0 < self.explained_energy_ratio <= 1.0:
            raise ValueError("explained_energy_ratio must be in (0, 1]")
        if self.max_replaced_basis <= 0:
            raise ValueError("max_replaced_basis must be positive")
        if self.orthogonality_tolerance <= 0:
            raise ValueError("orthogonality_tolerance must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BasisMaintenanceConfig:
        return cls(
            enabled=bool(value.get("enabled", True)),
            interval=int(value.get("interval", 10)),
            energy_mode=str(value.get("energy_mode", "relative")),
            residual_energy_threshold=float(value.get("residual_energy_threshold", 0.1)),
            residual_clip_norm=float(value.get("residual_clip_norm", 5.0)),
            center_update_rate=float(value.get("center_update_rate", 0.1)),
            explained_energy_ratio=float(value.get("explained_energy_ratio", 0.9)),
            max_replaced_basis=int(value.get("max_replaced_basis", 1)),
            usage_window=int(value.get("usage_window", 20)),
            orthogonality_tolerance=float(value.get("orthogonality_tolerance", 1.0e-5)),
            numerical_reorthogonalize=bool(value.get("numerical_reorthogonalize", True)),
        )


@dataclass(frozen=True, slots=True)
class BasisMaintenanceSummary:
    group_id: str
    triggered: bool
    residual_energy: float
    relative_residual_energy: float
    replaced_indices: tuple[int, ...]
    explained_energy: float
    center_update_norm: float
    cross_orthogonality_error: float = 0.0
    self_orthogonality_error: float = 0.0


def _query_weights(results: Sequence[ClientRoundResult]) -> Tensor:
    values = torch.tensor(
        [result.query_examples for result in results],
        dtype=torch.float32,
    )
    return values / values.sum()


def _uploaded_residual(result: ClientRoundResult) -> Tensor:
    if result.terminal_residual is None:
        raise ValueError("scheduled maintenance requires uploaded terminal residuals")
    return result.terminal_residual.detach().to(device="cpu", dtype=torch.float32)


def _orthogonality_errors(keep_matrix: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
    if keep_matrix.numel() == 0:
        cross = torch.zeros((), dtype=candidates.dtype, device=candidates.device)
    else:
        cross = torch.linalg.matrix_norm(keep_matrix.transpose(0, 1) @ candidates)
    identity = torch.eye(
        candidates.shape[1],
        dtype=candidates.dtype,
        device=candidates.device,
    )
    self_error = torch.linalg.matrix_norm(candidates.transpose(0, 1) @ candidates - identity)
    return cross, self_error


def _repair_new_directions(candidates: Tensor, keep_matrix: Tensor) -> Tensor:
    """Repair only new directions, never the retained basis."""

    repaired = candidates
    if keep_matrix.numel() > 0:
        repaired = repaired - keep_matrix @ (keep_matrix.transpose(0, 1) @ repaired)
    orthonormal, _ = torch.linalg.qr(repaired, mode="reduced")
    return orthonormal


class BasisMaintainer:
    """Maintain a group center and centered-residual shared directions."""

    def __init__(self, config: BasisMaintenanceConfig) -> None:
        self.config = config
        self._usage_history: dict[str, deque[Tensor]] = defaultdict(
            lambda: deque(maxlen=config.usage_window)
        )

    def is_scheduled(self, round_number: int) -> bool:
        if round_number <= 0:
            raise ValueError("round_number must be one-indexed and positive")
        return self.config.enabled and round_number % self.config.interval == 0

    def record_coordinate_usage(self, results: Sequence[ClientRoundResult]) -> None:
        if not self.config.enabled:
            return
        grouped: dict[str, list[ClientRoundResult]] = defaultdict(list)
        for result in results:
            grouped[result.group_id].append(result)
        for group_id, group_results in grouped.items():
            weights = _query_weights(group_results)
            usage = sum(
                float(weight) * result.terminal_coordinates.detach().float().cpu().square()
                for weight, result in zip(weights, group_results, strict=True)
            )
            self._usage_history[group_id].append(usage)

    def mean_coordinate_usage(self, group_id: str, *, num_basis: int) -> Tensor:
        history = self._usage_history.get(group_id)
        if not history:
            return torch.zeros(num_basis, dtype=torch.float32)
        usage = torch.stack(tuple(history), dim=0).mean(dim=0)
        if usage.shape != (num_basis,):
            raise ValueError(
                f"usage history for {group_id} has shape {usage.shape}, expected {(num_basis,)}"
            )
        return usage

    def residual_energy(self, results: Sequence[ClientRoundResult]) -> float:
        if not results:
            raise ValueError("group results cannot be empty")
        weights = _query_weights(results)
        return sum(
            float(weight) * result.residual_energy
            for weight, result in zip(weights, results, strict=True)
        )

    def relative_residual_energy(
        self,
        group: GroupState,
        results: Sequence[ClientRoundResult],
    ) -> float:
        weights = _query_weights(results)
        ratios = []
        for result in results:
            prompt_energy = result.prompt_energy
            if prompt_energy is None:
                residual = _uploaded_residual(result)
                prompt = group.subspace(
                    result.terminal_coordinates.to(group.subspace.center),
                    residual.to(group.subspace.center),
                )
                prompt_energy = float(prompt.float().square().sum())
            ratios.append(result.residual_energy / max(prompt_energy, 1.0e-12))
        return sum(
            float(weight) * ratio
            for weight, ratio in zip(weights, ratios, strict=True)
        )

    @torch.no_grad()
    def maintain_group(
        self,
        group: GroupState,
        results: Sequence[ClientRoundResult],
    ) -> BasisMaintenanceSummary:
        if not results:
            raise ValueError("group results cannot be empty")
        if any(result.group_id != group.group_id for result in results):
            raise ValueError("maintenance results must belong to one group")

        energy = self.residual_energy(results)
        relative_energy = self.relative_residual_energy(group, results)
        trigger_energy = (
            relative_energy if self.config.energy_mode == "relative" else energy
        )
        if (
            not self.config.enabled
            or trigger_energy < self.config.residual_energy_threshold
        ):
            return BasisMaintenanceSummary(
                group_id=group.group_id,
                triggered=False,
                residual_energy=energy,
                relative_residual_energy=relative_energy,
                replaced_indices=(),
                explained_energy=0.0,
                center_update_norm=0.0,
            )

        weights = _query_weights(results)
        residuals = [_uploaded_residual(result) for result in results]
        mean_residual = sum(
            float(weight) * residual for weight, residual in zip(weights, residuals, strict=True)
        )
        center_delta = self.config.center_update_rate * mean_residual
        group.subspace.set_center(
            group.subspace.center
            + center_delta.to(
                device=group.subspace.center.device,
                dtype=group.subspace.center.dtype,
            )
        )
        center_norm = float(torch.linalg.vector_norm(center_delta))

        # Centering is statistical and subtracts the full mean, independently
        # of the damped center update rate.
        centered = [residual - mean_residual for residual in residuals]
        residual_matrix = torch.stack(
            [
                weight.sqrt() * residual.reshape(-1)
                for weight, residual in zip(weights, centered, strict=True)
            ],
            dim=1,
        )
        left_vectors, singular_values, _ = torch.linalg.svd(
            residual_matrix,
            full_matrices=False,
        )
        if singular_values.numel() == 0:
            numerical_rank = 0
        else:
            tolerance = (
                max(residual_matrix.shape)
                * torch.finfo(singular_values.dtype).eps
                * float(singular_values.max())
            )
            numerical_rank = int((singular_values > tolerance).sum())
        if numerical_rank == 0:
            return BasisMaintenanceSummary(
                group_id=group.group_id,
                triggered=True,
                residual_energy=energy,
                relative_residual_energy=relative_energy,
                replaced_indices=(),
                explained_energy=0.0,
                center_update_norm=center_norm,
            )

        squared = singular_values[:numerical_rank].square()
        cumulative = squared.cumsum(dim=0) / squared.sum().clamp_min(1.0e-12)
        energy_count = (
            int(
                torch.nonzero(
                    cumulative >= self.config.explained_energy_ratio,
                    as_tuple=False,
                )[0]
            )
            + 1
        )
        requested = min(
            self.config.max_replaced_basis,
            group.num_basis,
            numerical_rank,
            energy_count,
        )
        usage = self.mean_coordinate_usage(group.group_id, num_basis=group.num_basis)
        replacement_indices = torch.argsort(usage, stable=True)[:requested]
        keep_mask = torch.ones(group.num_basis, dtype=torch.bool)
        keep_mask[replacement_indices] = False

        basis = group.subspace.basis.detach().float().cpu()
        keep_matrix = basis[keep_mask].reshape(int(keep_mask.sum()), -1).transpose(0, 1)
        candidates = left_vectors[:, :requested]
        cross_error, self_error = _orthogonality_errors(keep_matrix, candidates)
        if (
            self.config.numerical_reorthogonalize
            and max(float(cross_error), float(self_error)) > self.config.orthogonality_tolerance
        ):
            candidates = _repair_new_directions(candidates, keep_matrix)
            cross_error, self_error = _orthogonality_errors(keep_matrix, candidates)

        new_basis = basis.clone()
        new_basis[replacement_indices] = candidates.transpose(0, 1).reshape(
            requested,
            group.subspace.prompt_length,
            group.subspace.hidden_size,
        )
        group.subspace.set_basis(
            new_basis.to(
                device=group.subspace.basis.device,
                dtype=group.subspace.basis.dtype,
            ),
            orthonormalize=False,
        )

        explained = float(squared[:requested].sum() / squared.sum().clamp_min(1.0e-12))
        return BasisMaintenanceSummary(
            group_id=group.group_id,
            triggered=True,
            residual_energy=energy,
            relative_residual_energy=relative_energy,
            replaced_indices=tuple(int(index) for index in replacement_indices),
            explained_energy=explained,
            center_update_norm=center_norm,
            cross_orthogonality_error=float(cross_error),
            self_orthogonality_error=float(self_error),
        )

    def checkpoint_state(self) -> dict[str, list[Tensor]]:
        return {
            group_id: [value.clone() for value in history]
            for group_id, history in self._usage_history.items()
        }

    def load_checkpoint_state(self, state: Mapping[str, Sequence[Tensor]]) -> None:
        self._usage_history.clear()
        for group_id, values in state.items():
            history = deque(maxlen=self.config.usage_window)
            history.extend(value.detach().float().cpu() for value in values)
            self._usage_history[str(group_id)] = history
