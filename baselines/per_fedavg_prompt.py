"""First-order personalized FedAvg baseline for soft prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from baselines.local_prompt import (
    ClientPromptObjective,
    PromptOptimizationConfig,
    scalar_loss,
    validate_objectives,
)


@dataclass(frozen=True, slots=True)
class PerFedAvgConfig:
    inner: PromptOptimizationConfig
    server_learning_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.server_learning_rate <= 0:
            raise ValueError("server_learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class PerFedAvgRoundResult:
    global_prompt: Tensor
    adapted_prompts: dict[str, Tensor]
    mean_query_loss: float


def _adapt_for_meta_gradient(
    initial_prompt: Tensor,
    objective: ClientPromptObjective,
    config: PromptOptimizationConfig,
) -> Tensor:
    prompt = initial_prompt.detach().clone().requires_grad_(True)
    for step in range(config.steps):
        task_loss = scalar_loss("train_loss", objective.train_loss(prompt, step))
        penalty = 0.5 * config.weight_decay * prompt.square().sum()
        gradient = torch.autograd.grad(task_loss + penalty, prompt)[0]
        prompt = (
            prompt - config.learning_rate * gradient
        ).detach().requires_grad_(True)
    return prompt


class PerFedAvgPrompt(nn.Module):
    """Use query gradients after support adaptation to update one initialization.

    This prototype uses the common first-order Per-FedAvg/FOMAML approximation:
    it treats the adapted prompt as the global prompt when applying query gradients
    and therefore avoids a full prompt-sized Hessian graph.
    """

    def __init__(self, initial_prompt: Tensor) -> None:
        super().__init__()
        if initial_prompt.ndim != 2:
            raise ValueError("initial_prompt must have shape [prompt_length, hidden]")
        self.global_prompt = nn.Parameter(initial_prompt.detach().clone())

    def run_round(
        self,
        objectives: Sequence[ClientPromptObjective],
        config: PerFedAvgConfig,
    ) -> PerFedAvgRoundResult:
        values = validate_objectives(objectives)
        adapted: dict[str, Tensor] = {}
        gradients: list[tuple[Tensor, int]] = []
        query_losses: list[tuple[float, int]] = []
        start = self.global_prompt.detach()
        for objective in values:
            prompt = _adapt_for_meta_gradient(start, objective, config.inner)
            query_loss = scalar_loss("eval_loss", objective.eval_loss(prompt))
            query_gradient = torch.autograd.grad(query_loss, prompt)[0]
            adapted[objective.client_id] = prompt.detach()
            gradients.append((query_gradient.detach(), objective.num_examples))
            query_losses.append((float(query_loss.detach()), objective.num_examples))

        total_examples = sum(count for _, count in gradients)
        mean_gradient = sum(
            gradient * (count / total_examples) for gradient, count in gradients
        )
        with torch.no_grad():
            self.global_prompt.sub_(config.server_learning_rate * mean_gradient)
        mean_query_loss = sum(
            loss * count for loss, count in query_losses
        ) / total_examples
        return PerFedAvgRoundResult(
            global_prompt=self.global_prompt.detach().clone(),
            adapted_prompts=adapted,
            mean_query_loss=mean_query_loss,
        )
