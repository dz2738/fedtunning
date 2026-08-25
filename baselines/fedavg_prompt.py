"""FedAvg over full client soft prompts."""

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
    validate_objectives,
    weighted_prompt_average,
)


@dataclass(frozen=True, slots=True)
class FedAvgRoundResult:
    global_prompt: Tensor
    client_updates: tuple[PromptClientUpdate, ...]
    mean_train_loss: float


class FedAvgPrompt(nn.Module):
    """Broadcast one prompt, adapt locally, and average by example count."""

    def __init__(self, initial_prompt: Tensor) -> None:
        super().__init__()
        if initial_prompt.ndim != 2:
            raise ValueError("initial_prompt must have shape [prompt_length, hidden]")
        self.global_prompt = nn.Parameter(initial_prompt.detach().clone())

    def forward(self) -> Tensor:
        return self.global_prompt

    def run_round(
        self,
        objectives: Sequence[ClientPromptObjective],
        config: PromptOptimizationConfig,
    ) -> FedAvgRoundResult:
        values = validate_objectives(objectives)
        start = self.global_prompt.detach()
        updates = tuple(
            optimize_prompt(start, objective, config) for objective in values
        )
        average = weighted_prompt_average(updates)
        with torch.no_grad():
            self.global_prompt.copy_(average)
        return FedAvgRoundResult(
            global_prompt=self.global_prompt.detach().clone(),
            client_updates=updates,
            mean_train_loss=sum(update.mean_train_loss for update in updates)
            / len(updates),
        )
