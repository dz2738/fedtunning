"""IFCA-style hard client clustering with one soft prompt per cluster."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from baselines.local_prompt import (
    ClientPromptObjective,
    PromptClientUpdate,
    PromptOptimizationConfig,
    optimize_prompt,
    scalar_loss,
    validate_objectives,
    weighted_prompt_average,
)


@dataclass(frozen=True, slots=True)
class IFCARoundResult:
    assignments: dict[str, int]
    cluster_prompts: Tensor
    client_updates: tuple[PromptClientUpdate, ...]


class IFCAPrompt(nn.Module):
    """Assign by local training loss, then aggregate selected clusters."""

    def __init__(self, initial_prompts: Tensor) -> None:
        super().__init__()
        if initial_prompts.ndim != 3 or initial_prompts.shape[0] < 2:
            raise ValueError(
                "initial_prompts must have shape [clusters, prompt_length, hidden] "
                "with at least two clusters"
            )
        self.cluster_prompts = nn.Parameter(initial_prompts.detach().clone())

    @property
    def num_clusters(self) -> int:
        return self.cluster_prompts.shape[0]

    def assign(self, objective: ClientPromptObjective) -> int:
        losses = []
        for cluster_id in range(self.num_clusters):
            value = scalar_loss(
                "train_loss",
                objective.train_loss(self.cluster_prompts[cluster_id], 0),
            )
            losses.append(float(value.detach()))
        return min(range(self.num_clusters), key=lambda index: (losses[index], index))

    def run_round(
        self,
        objectives: Sequence[ClientPromptObjective],
        config: PromptOptimizationConfig,
    ) -> IFCARoundResult:
        values = validate_objectives(objectives)
        assignments = {objective.client_id: self.assign(objective) for objective in values}
        updates = tuple(
            optimize_prompt(
                self.cluster_prompts[assignments[objective.client_id]],
                objective,
                config,
            )
            for objective in values
        )
        by_cluster: dict[int, list[PromptClientUpdate]] = {
            cluster_id: [] for cluster_id in range(self.num_clusters)
        }
        for update in updates:
            by_cluster[assignments[update.client_id]].append(update)
        with torch.no_grad():
            for cluster_id, cluster_updates in by_cluster.items():
                if cluster_updates:
                    self.cluster_prompts[cluster_id].copy_(
                        weighted_prompt_average(cluster_updates)
                    )
        return IFCARoundResult(
            assignments=assignments,
            cluster_prompts=self.cluster_prompts.detach().clone(),
            client_updates=updates,
        )
