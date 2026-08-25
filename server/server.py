"""Single-process orchestration for the FedTaskPrompt server boundary."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from client.client import FederatedClient
from client.state import ClientRoundResult, ClientState, InnerLoopConfig
from model.coordinate_generator import CoordinateGenerator
from model.task_encoder import TaskSemanticEncoder
from server.aggregation import (
    AggregationSummary,
    ServerOptimizerConfig,
    SharedGeneratorTrainer,
)
from server.basis_maintenance import (
    BasisMaintainer,
    BasisMaintenanceConfig,
    BasisMaintenanceSummary,
)
from server.group_state import GroupState
from server.grouping import GroupingConfig, SemanticGrouper


@dataclass(frozen=True, slots=True)
class ServerRoundSummary:
    round_number: int
    client_ids: tuple[str, ...]
    aggregation: AggregationSummary
    maintenance: tuple[BasisMaintenanceSummary, ...]


class FedTaskPromptServer(nn.Module):
    """Own group prompt states and exactly one shared coordinate generator."""

    def __init__(
        self,
        *,
        task_encoder: TaskSemanticEncoder,
        coordinate_generator: CoordinateGenerator,
        grouping: GroupingConfig,
        optimizer: ServerOptimizerConfig,
        inner_loop: InnerLoopConfig,
        basis_maintenance: BasisMaintenanceConfig,
        prompt_length: int,
        hidden_size: int,
        num_basis: int,
        prompt_init_std: float = 0.02,
        projection_eps: float = 1.0e-6,
        prompt_dtype: torch.dtype | None = None,
        public_center: Tensor | None = None,
        public_basis: Tensor | None = None,
    ) -> None:
        super().__init__()
        if coordinate_generator.num_basis != num_basis:
            raise ValueError("coordinate generator and server use different basis counts")
        if coordinate_generator.embedding_dim != task_encoder.embedding_dim:
            raise ValueError("task encoder and coordinate generator dimensions differ")
        self.task_encoder = task_encoder
        self.generator_trainer = SharedGeneratorTrainer(coordinate_generator, optimizer)
        self.grouper = SemanticGrouper(grouping)
        self.inner_loop = inner_loop
        self.basis_maintainer = BasisMaintainer(basis_maintenance)
        self.prompt_length = int(prompt_length)
        self.hidden_size = int(hidden_size)
        self.num_basis = int(num_basis)
        self.prompt_init_std = float(prompt_init_std)
        self.projection_eps = float(projection_eps)
        self.prompt_dtype = prompt_dtype
        self.groups = nn.ModuleDict()
        self._task_embeddings: dict[str, Tensor] = {}
        self._round_number = 0
        self._public_center = None if public_center is None else public_center.detach().clone()
        self._public_basis = None if public_basis is None else public_basis.detach().clone()

    @property
    def round_number(self) -> int:
        return self._round_number

    @property
    def task_embeddings(self) -> dict[str, Tensor]:
        return {
            client_id: value.clone()
            for client_id, value in self._task_embeddings.items()
        }

    def initialize_groups_from_embeddings(
        self,
        clients: Mapping[str, ClientState],
        embeddings: Mapping[str, Tensor],
    ) -> dict[str, str]:
        """Initialize semantic groups from precomputed task embeddings."""

        if set(clients) != set(embeddings):
            raise ValueError(
                "client IDs and task-embedding IDs must match"
            )

        self._task_embeddings = {
            client_id: embedding.detach().float().cpu()
            for client_id, embedding in embeddings.items()
        }

        assignments = self.grouper.fit(
            self._task_embeddings
        )

        parameter = next(
            self.generator_trainer.generator.parameters()
        )

        self.groups = nn.ModuleDict()

        for group_id in self.grouper.group_ids:
            group = GroupState(
                group_id=group_id,
                semantic_prototype=self.grouper.centroid(group_id),
                prompt_length=self.prompt_length,
                hidden_size=self.hidden_size,
                num_basis=self.num_basis,
                prompt_init_std=self.prompt_init_std,
                projection_eps=self.projection_eps,
                device=parameter.device,
                dtype=self.prompt_dtype,
            )

            group.set_members(
                self.grouper.members(group_id)
            )

            group.initialize_prompt_subspace(
                center=self._public_center,
                basis=self._public_basis,
            )

            self.groups[group_id] = group

        for client_id, group_id in assignments.items():
            clients[client_id].assign_group(group_id)

        return assignments

    def initialize_groups(
        self,
        clients: Mapping[str, ClientState],
    ) -> dict[str, str]:
        if not clients:
            raise ValueError("clients cannot be empty")

        client_ids = tuple(sorted(clients))

        descriptions = [
            clients[client_id].description
            for client_id in client_ids
        ]

        encoded = (
            self.task_encoder
            .encode_descriptions(descriptions)
            .detach()
            .float()
            .cpu()
        )

        embeddings = {
            client_id: embedding
            for client_id, embedding in zip(
                client_ids,
                encoded,
                strict=True,
            )
        }

        return self.initialize_groups_from_embeddings(
            clients,
            embeddings,
        )

    def initial_coordinates(self, client_id: str) -> Tensor:
        if client_id not in self._task_embeddings:
            raise KeyError(f"unknown client {client_id!r}; initialize groups first")
        group_id = self.grouper.assignments[client_id]
        return self.generator_trainer.initial_coordinates(
            self._task_embeddings[client_id],
            self.groups[group_id],
        )

    def run_round(
        self,
        *,
        clients: Mapping[str, FederatedClient],
        selected_client_ids: Sequence[str],
        seed: int,
    ) -> ServerRoundSummary:
        selected = tuple(str(client_id) for client_id in selected_client_ids)
        if not selected:
            raise ValueError("selected_client_ids cannot be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("selected_client_ids contains duplicates")
        missing = [client_id for client_id in selected if client_id not in clients]
        if missing:
            raise KeyError(f"unknown selected clients: {missing}")
        if not self.groups:
            raise RuntimeError("initialize_groups must be called before run_round")

        next_round = self._round_number + 1
        maintenance_round = self.basis_maintainer.is_scheduled(next_round)
        results: list[ClientRoundResult] = []
        for position, client_id in enumerate(selected):
            group_id = self.grouper.assignments[client_id]
            group = self.groups[group_id]
            initial = self.generator_trainer.initial_coordinates(
                self._task_embeddings[client_id],
                group,
            )
            results.append(
                clients[client_id].run_round(
                    group_id=group_id,
                    initial_coordinates=initial,
                    subspace=group.subspace,
                    config=self.inner_loop,
                    round_seed=seed + next_round * 100_003 + position,
                    return_residual=maintenance_round,
                )
            )

        aggregation = self.generator_trainer.update(
            results=results,
            task_embeddings=self._task_embeddings,
            groups=self.groups,
        )
        self.basis_maintainer.record_coordinate_usage(results)

        maintenance_summaries: list[BasisMaintenanceSummary] = []
        if maintenance_round:
            grouped: dict[str, list[ClientRoundResult]] = defaultdict(list)
            for result in results:
                grouped[result.group_id].append(result)
            for group_id in sorted(grouped):
                maintenance_summaries.append(
                    self.basis_maintainer.maintain_group(
                        self.groups[group_id],
                        grouped[group_id],
                    )
                )

        self._round_number = next_round
        return ServerRoundSummary(
            round_number=next_round,
            client_ids=selected,
            aggregation=aggregation,
            maintenance=tuple(maintenance_summaries),
        )

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "round_number": self._round_number,
            "module": self.state_dict(),
            "optimizer": self.generator_trainer.optimizer.state_dict(),
            "task_embeddings": self.task_embeddings,
            "assignments": self.grouper.assignments,
            "basis_usage": self.basis_maintainer.checkpoint_state(),
        }

    def load_checkpoint_state(
        self,
        state: Mapping[str, object],
    ) -> None:
        """Restore a checkpoint after groups are reconstructed."""

        saved_assignments = {
            str(client_id): str(group_id)
            for client_id, group_id
            in dict(state["assignments"]).items()
        }

        if self.grouper.assignments != saved_assignments:
            raise ValueError(
                "checkpoint groups differ from initialized groups; "
                "use the same clients, task descriptions, "
                "and grouping configuration"
            )

        self.load_state_dict(
            state["module"]
        )

        self.task_encoder.clear_cache()

        self.generator_trainer.optimizer.load_state_dict(
            state["optimizer"]
        )

        self._task_embeddings = {
            str(client_id): embedding.detach().float().cpu()
            for client_id, embedding
            in dict(state["task_embeddings"]).items()
        }

        self.basis_maintainer.load_checkpoint_state(
            state["basis_usage"]
        )

        self._round_number = int(
            state["round_number"]
        )
