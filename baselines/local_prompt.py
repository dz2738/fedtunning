"""Independent local soft-prompt training baseline.

The baseline classes never own a language model.  A caller constructs the
loss callbacks with the process-wide ``SharedBackbone`` and passes them here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


PromptTrainLoss = Callable[[Tensor, int], Tensor]
PromptEvalLoss = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class ClientPromptObjective:
    """One client's support/train and query/evaluation objectives."""

    client_id: str
    train_loss: PromptTrainLoss
    eval_loss: PromptEvalLoss
    num_examples: int

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id cannot be empty")
        if self.num_examples <= 0:
            raise ValueError("num_examples must be positive")


@dataclass(frozen=True, slots=True)
class PromptOptimizationConfig:
    steps: int = 5
    learning_rate: float = 0.1
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True, slots=True)
class PromptClientUpdate:
    client_id: str
    prompt: Tensor
    mean_train_loss: float
    eval_loss: float
    num_examples: int


def scalar_loss(name: str, value: Tensor) -> Tensor:
    """Validate a loss callback result before differentiating it."""

    if value.ndim != 0:
        raise ValueError(f"{name} must return a scalar tensor, got {value.shape}")
    if not torch.isfinite(value):
        raise FloatingPointError(f"{name} returned a non-finite value")
    return value


def validate_objectives(
    objectives: Sequence[ClientPromptObjective],
) -> tuple[ClientPromptObjective, ...]:
    values = tuple(objectives)
    if not values:
        raise ValueError("at least one client objective is required")
    client_ids = [objective.client_id for objective in values]
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("client objectives must have unique client IDs")
    return values


def optimize_prompt(
    initial_prompt: Tensor,
    objective: ClientPromptObjective,
    config: PromptOptimizationConfig,
) -> PromptClientUpdate:
    """Run detached local SGD on a prompt without retaining a backbone graph."""

    if initial_prompt.ndim != 2:
        raise ValueError("initial_prompt must have shape [prompt_length, hidden]")
    prompt = initial_prompt.detach().clone().requires_grad_(True)
    losses: list[float] = []
    for step in range(config.steps):
        task_loss = scalar_loss("train_loss", objective.train_loss(prompt, step))
        regularizer = 0.5 * config.weight_decay * prompt.square().sum()
        gradient = torch.autograd.grad(task_loss + regularizer, prompt)[0]
        prompt = (
            prompt - config.learning_rate * gradient
        ).detach().requires_grad_(True)
        losses.append(float(task_loss.detach()))

    evaluation = scalar_loss("eval_loss", objective.eval_loss(prompt))
    return PromptClientUpdate(
        client_id=objective.client_id,
        prompt=prompt.detach(),
        mean_train_loss=sum(losses) / len(losses),
        eval_loss=float(evaluation.detach()),
        num_examples=objective.num_examples,
    )


def weighted_prompt_average(updates: Sequence[PromptClientUpdate]) -> Tensor:
    """Example-weighted mean of compatible client prompts."""

    values = tuple(updates)
    if not values:
        raise ValueError("cannot average an empty update sequence")
    reference = values[0].prompt
    if any(update.prompt.shape != reference.shape for update in values):
        raise ValueError("all client prompts must have the same shape")
    total_examples = sum(update.num_examples for update in values)
    return sum(
        update.prompt.to(device=reference.device, dtype=reference.dtype)
        * (update.num_examples / total_examples)
        for update in values
    )


class LocalPrompt:
    """Keep a separate lightweight prompt tensor for every client."""

    def __init__(self, initial_prompt: Tensor) -> None:
        if initial_prompt.ndim != 2:
            raise ValueError("initial_prompt must have shape [prompt_length, hidden]")
        self._initial_prompt = initial_prompt.detach().clone()
        self._prompts: dict[str, Tensor] = {}

    def prompt(self, client_id: str) -> Tensor:
        if not client_id:
            raise ValueError("client_id cannot be empty")
        if client_id not in self._prompts:
            self._prompts[client_id] = self._initial_prompt.clone()
        return self._prompts[client_id]

    def run_client(
        self,
        objective: ClientPromptObjective,
        config: PromptOptimizationConfig,
    ) -> PromptClientUpdate:
        update = optimize_prompt(self.prompt(objective.client_id), objective, config)
        self._prompts[objective.client_id] = update.prompt.detach().clone()
        return update

    def state_dict(self) -> dict[str, object]:
        return {
            "initial_prompt": self._initial_prompt.detach().cpu(),
            "client_prompts": {
                client_id: prompt.detach().cpu()
                for client_id, prompt in self._prompts.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        initial = state.get("initial_prompt")
        prompts = state.get("client_prompts")
        if not isinstance(initial, Tensor) or not isinstance(prompts, Mapping):
            raise TypeError("invalid LocalPrompt state")
        if initial.ndim != 2:
            raise ValueError("saved initial prompt must be a matrix")
        restored: dict[str, Tensor] = {}
        for client_id, prompt in prompts.items():
            if not isinstance(client_id, str) or not isinstance(prompt, Tensor):
                raise TypeError("invalid client prompt entry")
            if prompt.shape != initial.shape:
                raise ValueError("saved client prompt shape mismatch")
            restored[client_id] = prompt.detach().clone()
        self._initial_prompt = initial.detach().clone()
        self._prompts = restored
