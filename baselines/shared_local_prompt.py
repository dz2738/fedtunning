"""Prompt baseline with globally shared and client-private token slots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
class SharedLocalRoundResult:
    shared_prompt: Tensor
    client_updates: tuple[PromptClientUpdate, ...]


class SharedLocalPrompt(nn.Module):
    """Aggregate shared prefix slots while retaining private suffix slots."""

    def __init__(
        self,
        initial_prompt: Tensor,
        *,
        shared_length: int,
    ) -> None:
        super().__init__()
        if initial_prompt.ndim != 2:
            raise ValueError("initial_prompt must have shape [prompt_length, hidden]")
        if not 0 < shared_length < initial_prompt.shape[0]:
            raise ValueError("shared_length must split the prompt into two nonempty parts")
        self.shared_prompt = nn.Parameter(initial_prompt[:shared_length].detach().clone())
        self.register_buffer(
            "initial_local_prompt",
            initial_prompt[shared_length:].detach().clone(),
        )
        self._local_prompts: dict[str, Tensor] = {}

    def local_prompt(self, client_id: str) -> Tensor:
        if not client_id:
            raise ValueError("client_id cannot be empty")
        if client_id not in self._local_prompts:
            self._local_prompts[client_id] = self.initial_local_prompt.detach().clone()
        local = self._local_prompts[client_id]
        return local.to(
            device=self.shared_prompt.device,
            dtype=self.shared_prompt.dtype,
        )

    def prompt(self, client_id: str) -> Tensor:
        return torch.cat((self.shared_prompt, self.local_prompt(client_id)), dim=0)

    def run_round(
        self,
        objectives: Sequence[ClientPromptObjective],
        config: PromptOptimizationConfig,
    ) -> SharedLocalRoundResult:
        values = validate_objectives(objectives)
        updates = tuple(
            optimize_prompt(self.prompt(objective.client_id), objective, config)
            for objective in values
        )
        split = self.shared_prompt.shape[0]
        shared_updates: list[PromptClientUpdate] = []
        for update in updates:
            shared_updates.append(
                PromptClientUpdate(
                    client_id=update.client_id,
                    prompt=update.prompt[:split],
                    mean_train_loss=update.mean_train_loss,
                    eval_loss=update.eval_loss,
                    num_examples=update.num_examples,
                )
            )
            self._local_prompts[update.client_id] = update.prompt[split:].detach().cpu()
        with torch.no_grad():
            self.shared_prompt.copy_(weighted_prompt_average(shared_updates))
        return SharedLocalRoundResult(
            shared_prompt=self.shared_prompt.detach().clone(),
            client_updates=updates,
        )

    def private_state_dict(self) -> dict[str, Tensor]:
        return {
            client_id: prompt.detach().cpu()
            for client_id, prompt in self._local_prompts.items()
        }

    def load_private_state_dict(self, state: Mapping[str, Tensor]) -> None:
        expected_shape = self.initial_local_prompt.shape
        restored: dict[str, Tensor] = {}
        for client_id, prompt in state.items():
            if prompt.shape != expected_shape:
                raise ValueError(f"private prompt shape mismatch for {client_id!r}")
            restored[client_id] = prompt.detach().cpu().clone()
        self._local_prompts = restored
