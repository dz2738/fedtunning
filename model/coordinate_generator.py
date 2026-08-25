"""Group-specific mapping from task semantics to initial prompt coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CoordinateGeneratorConfig:
    embedding_dim: int
    num_basis: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    output_activation: str = "identity"

    def __post_init__(self) -> None:
        if min(self.embedding_dim, self.num_basis, self.hidden_dim) <= 0:
            raise ValueError("embedding_dim, num_basis, and hidden_dim must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.output_activation not in {"identity", "tanh"}:
            raise ValueError("output_activation must be 'identity' or 'tanh'")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        embedding_dim: int,
        num_basis: int,
    ) -> CoordinateGeneratorConfig:
        return cls(
            embedding_dim=embedding_dim,
            num_basis=num_basis,
            hidden_dim=int(value.get("hidden_dim", 256)),
            num_layers=int(value.get("num_layers", 2)),
            dropout=float(value.get("dropout", 0.1)),
            output_activation=str(value.get("output_activation", "identity")),
        )


class CoordinateGenerator(nn.Module):
    """Implement c_n^(0) = G_theta(z_n, mu_r).

    The prototype concatenates the two vectors exactly once and deliberately
    avoids adding unreported similarity or interaction features.
    """

    def __init__(self, config: CoordinateGeneratorConfig) -> None:
        super().__init__()
        self.settings = config
        layers: list[nn.Module] = []
        input_dim = config.embedding_dim * 2
        for _ in range(config.num_layers):
            layers.extend(
                (
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )
            input_dim = config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(config.hidden_dim, config.num_basis)
        self.output_activation: nn.Module
        if config.output_activation == "identity":
            self.output_activation = nn.Identity()
        else:
            self.output_activation = nn.Tanh()
        self.reset_parameters()

    @property
    def embedding_dim(self) -> int:
        return self.settings.embedding_dim

    @property
    def num_basis(self) -> int:
        return self.settings.num_basis

    def reset_parameters(self) -> None:
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # A small final layer keeps initial coordinates close to the shared
        # center while still allowing gradients to reach the entire MLP.
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, task_embedding: Tensor, group_embedding: Tensor) -> Tensor:
        if task_embedding.ndim < 1 or group_embedding.ndim < 1:
            raise ValueError("task_embedding and group_embedding must have at least one dimension")
        if task_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"task embedding must end in {self.embedding_dim}, got {task_embedding.shape}"
            )
        if group_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"group embedding must end in {self.embedding_dim}, got {group_embedding.shape}"
            )
        if task_embedding.device != group_embedding.device:
            raise ValueError("task and group embeddings must share a device")
        if task_embedding.dtype != group_embedding.dtype:
            raise ValueError("task and group embeddings must share a dtype")

        task_embedding, group_embedding = torch.broadcast_tensors(
            task_embedding,
            group_embedding,
        )
        condition = torch.cat((task_embedding, group_embedding), dim=-1)
        return self.output_activation(self.output(self.trunk(condition)))


def coordinate_l2_penalty(coordinates: Tensor, *, coefficient: float) -> Tensor:
    """Return lambda/2 times the batch-mean squared coordinate norm."""

    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    if coordinates.ndim < 1:
        raise ValueError("coordinates must have at least one dimension")
    squared_norm = coordinates.square().sum(dim=-1)
    return 0.5 * coefficient * squared_norm.mean()

