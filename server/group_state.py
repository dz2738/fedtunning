"""Group-specific prompt state without a group-specific coordinate generator."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.coordinate_generator import CoordinateGenerator
from model.prompt_subspace import PromptSubspace


class GroupState(nn.Module):
    """Store one group's semantic prototype and affine prompt subspace.

    The coordinate generator deliberately is not a child module. The server owns
    one global ``CoordinateGenerator`` and passes it into ``initial_coordinates``.
    """

    def __init__(
        self,
        *,
        group_id: str,
        semantic_prototype: Tensor,
        prompt_length: int,
        hidden_size: int,
        num_basis: int,
        prompt_init_std: float = 0.02,
        projection_eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        group_id = str(group_id).strip()
        if not group_id:
            raise ValueError("group_id must be non-empty")
        self.group_id = group_id
        self.subspace = PromptSubspace(
            prompt_length=prompt_length,
            hidden_size=hidden_size,
            num_basis=num_basis,
            init_std=prompt_init_std,
            projection_eps=projection_eps,
            device=device,
            dtype=dtype,
        )
        prototype = self._normalized_prototype(semantic_prototype, device=device)
        self.register_buffer("semantic_prototype", prototype)
        self._members: set[str] = set()

    @staticmethod
    def _normalized_prototype(
        value: Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if value.ndim != 1 or not value.is_floating_point():
            raise ValueError("semantic_prototype must be a floating-point vector")
        prototype = value.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(prototype).all() or float(prototype.norm()) == 0.0:
            raise ValueError("semantic_prototype must be finite and non-zero")
        return F.normalize(prototype, p=2, dim=0)

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted(self._members))

    @property
    def num_basis(self) -> int:
        return self.subspace.num_basis

    def set_members(self, client_ids: Iterable[str]) -> None:
        members = {str(client_id).strip() for client_id in client_ids}
        if "" in members:
            raise ValueError("client IDs must be non-empty")
        self._members = members

    @torch.no_grad()
    def set_semantic_prototype(self, value: Tensor) -> None:
        prototype = self._normalized_prototype(
            value,
            device=self.semantic_prototype.device,
        )
        if prototype.shape != self.semantic_prototype.shape:
            raise ValueError(
                f"expected semantic prototype shape {self.semantic_prototype.shape}, "
                f"got {prototype.shape}"
            )
        self.semantic_prototype.copy_(prototype)

    def initial_coordinates(
        self,
        generator: CoordinateGenerator,
        task_embedding: Tensor,
    ) -> Tensor:
        if generator.num_basis != self.num_basis:
            raise ValueError("global generator and group subspace use different basis counts")
        parameter = next(generator.parameters())
        task_embedding = task_embedding.to(device=parameter.device, dtype=parameter.dtype)
        prototype = self.semantic_prototype.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return generator(task_embedding, prototype)

    def get_extra_state(self) -> dict[str, Any]:
        return {"group_id": self.group_id, "members": list(self.members)}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.group_id = str(state["group_id"])
        self.set_members(state.get("members", ()))

    @torch.no_grad()
    def initialize_prompt_subspace(
        self,
        *,
        center: Tensor | None = None,
        basis: Tensor | None = None,
    ) -> None:
        """Initialize a group from the public prompt subspace."""

        if center is not None:
            self.subspace.set_center(center)

        if basis is not None:
            self.subspace.set_basis(
                basis,
                orthonormalize=True,
            )
