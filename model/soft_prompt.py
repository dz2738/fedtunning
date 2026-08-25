"""Soft-prompt parameters and encoder-input injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


PromptInit = Literal["normal", "zeros", "vocab"]


def _validate_token_inputs(token_embeddings: Tensor, attention_mask: Tensor) -> None:
    if token_embeddings.ndim != 3:
        raise ValueError(
            "token_embeddings must have shape [batch, sequence, hidden], "
            f"got {tuple(token_embeddings.shape)}"
        )
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, sequence], got {attention_mask.shape}"
        )
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} does not match "
            f"token embeddings {token_embeddings.shape[:2]}"
        )
    if attention_mask.device != token_embeddings.device:
        raise ValueError("attention_mask and token_embeddings must share a device")


def expand_prompt(prompt_embeddings: Tensor, batch_size: int) -> Tensor:
    """Broadcast a shared prompt or validate a client-batched prompt."""

    if prompt_embeddings.ndim == 2:
        return prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    if prompt_embeddings.ndim != 3:
        raise ValueError(
            "prompt_embeddings must have shape [prompt_length, hidden] or "
            f"[batch, prompt_length, hidden], got {tuple(prompt_embeddings.shape)}"
        )
    if prompt_embeddings.shape[0] != batch_size:
        raise ValueError(
            f"prompt batch size {prompt_embeddings.shape[0]} does not match {batch_size}"
        )
    return prompt_embeddings


@dataclass(frozen=True, slots=True)
class PromptedInputs:
    inputs_embeds: Tensor
    attention_mask: Tensor
    prompt_length: int


def prepend_soft_prompt(
    token_embeddings: Tensor,
    attention_mask: Tensor,
    prompt_embeddings: Tensor,
) -> PromptedInputs:
    """Prepend continuous embeddings and matching visible attention positions.

    For encoder-decoder models, labels describe decoder tokens and therefore do
    not need to be shifted when a prompt is added to the encoder sequence.
    """

    _validate_token_inputs(token_embeddings, attention_mask)
    prompt = expand_prompt(prompt_embeddings, token_embeddings.shape[0])
    if prompt.shape[-1] != token_embeddings.shape[-1]:
        raise ValueError(
            f"prompt hidden size {prompt.shape[-1]} does not match "
            f"token hidden size {token_embeddings.shape[-1]}"
        )
    prompt = prompt.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    prompt_mask = torch.ones(
        prompt.shape[:2],
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return PromptedInputs(
        inputs_embeds=torch.cat((prompt, token_embeddings), dim=1),
        attention_mask=torch.cat((prompt_mask, attention_mask), dim=1),
        prompt_length=prompt.shape[1],
    )


class SoftPromptInjector(nn.Module):
    """Use one shared token embedding table without owning client prompts."""

    def __init__(self, input_embeddings: nn.Embedding) -> None:
        super().__init__()
        # The backbone already owns this module. Bypass nn.Module registration
        # here to avoid duplicate state_dict entries for the same weight.
        object.__setattr__(self, "_input_embeddings", input_embeddings)

    @property
    def input_embeddings(self) -> nn.Embedding:
        return self._input_embeddings

    @property
    def hidden_size(self) -> int:
        return self.input_embeddings.embedding_dim

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        prompt_embeddings: Tensor,
    ) -> PromptedInputs:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, sequence], got {input_ids.shape}")
        token_embeddings = self.input_embeddings(input_ids)
        return prepend_soft_prompt(token_embeddings, attention_mask, prompt_embeddings)


def initialize_prompt_(
    prompt: Tensor,
    *,
    strategy: PromptInit,
    init_std: float,
    token_embedding_weight: Tensor | None = None,
) -> None:
    """Initialize a prompt in-place without keeping a token-table reference."""

    if init_std <= 0:
        raise ValueError("init_std must be positive")
    with torch.no_grad():
        if strategy == "normal":
            nn.init.normal_(prompt, mean=0.0, std=init_std)
        elif strategy == "zeros":
            prompt.zero_()
        elif strategy == "vocab":
            if token_embedding_weight is None:
                raise ValueError("vocab initialization requires token_embedding_weight")
            if token_embedding_weight.ndim != 2:
                raise ValueError("token_embedding_weight must have shape [vocab, hidden]")
            if token_embedding_weight.shape[1] != prompt.shape[1]:
                raise ValueError("token embedding hidden size does not match prompt")
            indices = torch.arange(prompt.shape[0], device=token_embedding_weight.device)
            indices = indices.remainder(token_embedding_weight.shape[0])
            prompt.copy_(
                token_embedding_weight.index_select(0, indices).to(
                    device=prompt.device,
                    dtype=prompt.dtype,
                )
            )
        else:
            raise ValueError(f"unknown prompt initialization strategy: {strategy!r}")


class TrainableSoftPrompt(nn.Module):
    """Standalone prompt used by local and FedAvg baselines."""

    def __init__(
        self,
        *,
        prompt_length: int,
        hidden_size: int,
        init: PromptInit = "normal",
        init_std: float = 0.02,
        token_embedding_weight: Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if prompt_length <= 0 or hidden_size <= 0:
            raise ValueError("prompt_length and hidden_size must be positive")
        self.embedding = nn.Parameter(
            torch.empty(prompt_length, hidden_size, device=device, dtype=dtype)
        )
        initialize_prompt_(
            self.embedding,
            strategy=init,
            init_std=init_std,
            token_embedding_weight=token_embedding_weight,
        )

    @property
    def prompt_length(self) -> int:
        return self.embedding.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.embedding.shape[1]

    def forward(self, batch_size: int | None = None) -> Tensor:
        if batch_size is None:
            return self.embedding
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)
