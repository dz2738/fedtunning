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
    interval: int = 20
    residual_energy_threshold: float = 1.0
    residual_clip_norm: float = 5.0
    center_update_rate: float = 0.1
    explained_energy_ratio: float = 0.9
    max_replaced_basis: int = 2
    usage_window: int = 20
    generator_output_reset: str = "none"

    def __post_init__(self) -> None:
        if self.interval <= 0 or self.usage_window <= 0:
            raise ValueError("maintenance interval and usage_window must be positive")
        if self.residual_energy_threshold < 0 or self.residual_clip_norm <= 0:
            raise ValueError("residual energy must be non-negative and clip norm positive")
        if not 0.0 <= self.center_update_rate <= 1.0:
            raise ValueError("center_update_rate must be in [0, 1]")
        if not 0.0 < self.explained_energy_ratio <= 1.0:
            raise ValueError("explained_energy_ratio must be in (0, 1]")
        if self.max_replaced_basis <= 0:
            raise ValueError("max_replaced_basis must be positive")
        if self.generator_output_reset != "none":
            raise ValueError(
                "a group-local basis replacement cannot reset rows of the global generator"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BasisMaintenanceConfig:
        return cls(
            enabled=bool(value.get("enabled", True)),
            interval=int(value.get("interval", 20)),
            residual_energy_threshold=float(
                value.get("residual_energy_threshold", 1.0)
            ),
            residual_clip_norm=float(value.get("residual_clip_norm", 5.0)),
            center_update_rate=float(value.get("center_update_rate", 0.1)),
            explained_energy_ratio=float(value.get("explained_energy_ratio", 0.9)),
            max_replaced_basis=int(value.get("max_replaced_basis", 2)),
            usage_window=int(value.get("usage_window", 20)),
            generator_output_reset=str(value.get("generator_output_reset", "none")),
        )


@dataclass(frozen=True, slots=True)
class BasisMaintenanceSummary:
    group_id: str
    triggered: bool
    residual_energy: float
    replaced_indices: tuple[int, ...]
    explained_energy: float
    center_update_norm: float


def _query_weights(results: Sequence[ClientRoundResult]) -> Tensor:
    values = torch.tensor(
        [result.query_examples for result in results],
        dtype=torch.float32,
    )
    return values / values.sum()


def _clip_residual(residual: Tensor, max_norm: float) -> Tensor:
    value = residual.detach().to(device="cpu", dtype=torch.float32)
    norm = torch.linalg.vector_norm(value)
    scale = torch.clamp(max_norm / norm.clamp_min(1.0e-12), max=1.0)
    return value * scale


class BasisMaintainer:
    """Implement Eqs. (37)--(48) without mutating the global generator."""

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
                f"usage history for {group_id} has shape {usage.shape}, "
                f"expected {(num_basis,)}"
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
        if not self.config.enabled or energy < self.config.residual_energy_threshold:
            return BasisMaintenanceSummary(
                group_id=group.group_id,
                triggered=False,
                residual_energy=energy,
                replaced_indices=(),
                explained_energy=0.0,
                center_update_norm=0.0,
            )
        if any(result.terminal_residual is None for result in results):
            raise ValueError("scheduled maintenance requires uploaded terminal residuals")

        weights = _query_weights(results)
        residuals = [
            _clip_residual(result.terminal_residual, self.config.residual_clip_norm)
            for result in results
        ]
        center_update = sum(
            float(weight) * residual
            for weight, residual in zip(weights, residuals, strict=True)
        )
        center_delta = self.config.center_update_rate * center_update
        group.subspace.set_center(
            group.subspace.center + center_delta.to(
                device=group.subspace.center.device,
                dtype=group.subspace.center.dtype,
            )
        )

        residual_matrix = torch.stack(
            [
                weight.sqrt() * residual.reshape(-1)
                for weight, residual in zip(weights, residuals, strict=True)
            ],
            dim=1,
        )
        left_vectors, singular_values, _ = torch.linalg.svd(
            residual_matrix,
            full_matrices=False,
        )
        squared = singular_values.square()
        numerical_rank = int((squared > torch.finfo(squared.dtype).eps).sum())
        if numerical_rank == 0:
            return BasisMaintenanceSummary(
                group_id=group.group_id,
                triggered=True,
                residual_energy=energy,
                replaced_indices=(),
                explained_energy=0.0,
                center_update_norm=float(torch.linalg.vector_norm(center_delta)),
            )

        cumulative = squared.cumsum(dim=0) / squared.sum().clamp_min(1.0e-12)
        energy_count = int(
            torch.nonzero(
                cumulative >= self.config.explained_energy_ratio,
                as_tuple=False,
            )[0]
        ) + 1
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
        if keep_matrix.numel() > 0:
            candidates = candidates - keep_matrix @ (keep_matrix.transpose(0, 1) @ candidates)

        candidate_rank = int(torch.linalg.matrix_rank(candidates))
        actual = min(requested, candidate_rank)
        if actual == 0:
            return BasisMaintenanceSummary(
                group_id=group.group_id,
                triggered=True,
                residual_energy=energy,
                replaced_indices=(),
                explained_energy=0.0,
                center_update_norm=float(torch.linalg.vector_norm(center_delta)),
            )
        replacement_indices = replacement_indices[:actual]
        orthonormal, _ = torch.linalg.qr(candidates[:, :actual], mode="reduced")
        new_basis = basis.clone()
        new_basis[replacement_indices] = orthonormal.transpose(0, 1).reshape(
            actual,
            group.subspace.prompt_length,
            group.subspace.hidden_size,
        )
        group.subspace.set_basis(
            new_basis.to(
                device=group.subspace.basis.device,
                dtype=group.subspace.basis.dtype,
            ),
            # Kept rows and QR candidates are already mutually orthonormal.
            # Re-running QR over the full matrix would rotate directions that
            # the paper explicitly keeps unchanged.
            orthonormalize=False,
        )

        explained = float(squared[:actual].sum() / squared.sum().clamp_min(1.0e-12))
        return BasisMaintenanceSummary(
            group_id=group.group_id,
            triggered=True,
            residual_energy=energy,
            replaced_indices=tuple(int(index) for index in replacement_indices),
            explained_energy=explained,
            center_update_norm=float(torch.linalg.vector_norm(center_delta)),
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
