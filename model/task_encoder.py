"""Task-description embeddings produced by the single shared encoder."""

from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.schema import TaskDescription
from model.backbone import SharedBackbone


@dataclass(frozen=True, slots=True)
class TaskEncoderConfig:
    source: str = "shared_encoder"
    pooling: str = "masked_mean"
    embedding_dim: int = 256
    projection_trainable: bool = False
    normalize: bool = True
    max_length: int = 128

    def __post_init__(self) -> None:
        if self.source != "shared_encoder":
            raise ValueError("the prototype supports only source='shared_encoder'")
        if self.pooling != "masked_mean":
            raise ValueError("the prototype supports only pooling='masked_mean'")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskEncoderConfig:
        return cls(
            source=str(value.get("source", "shared_encoder")),
            pooling=str(value.get("pooling", "masked_mean")),
            embedding_dim=int(value.get("embedding_dim", 256)),
            projection_trainable=bool(value.get("projection_trainable", False)),
            normalize=bool(value.get("normalize", True)),
            max_length=int(value.get("max_length", 128)),
        )


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool token states without allowing padded positions to affect the mean."""

    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, sequence, hidden], got {hidden_states.shape}"
        )
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask shape {attention_mask.shape} does not match "
            f"hidden states {hidden_states.shape[:2]}"
        )
    mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    token_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / token_count


class TaskSemanticEncoder(nn.Module):
    """Encode task and group descriptions without owning another backbone.

    A weak reference is used deliberately: registering SharedBackbone as a child
    here would duplicate its keys whenever a server checkpoint is created.
    """

    def __init__(
        self,
        backbone: SharedBackbone,
        config: TaskEncoderConfig,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_backbone_ref", weakref.ref(backbone))
        self.settings = config
        self.projection = nn.Linear(
            backbone.hidden_size,
            config.embedding_dim,
            bias=True,
            device=backbone.runtime_device,
            dtype=torch.float32,
        )
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection.requires_grad_(config.projection_trainable)
        self._cache: dict[str, Tensor] = {}

    @property
    def backbone(self) -> SharedBackbone:
        backbone = self._backbone_ref()
        if backbone is None:
            raise RuntimeError("the shared backbone was released before the task encoder")
        return backbone

    @property
    def embedding_dim(self) -> int:
        return self.projection.out_features

    @property
    def cache_enabled(self) -> bool:
        return not self.settings.projection_trainable

    def clear_cache(self) -> None:
        self._cache.clear()

    def train(self, mode: bool = True) -> TaskSemanticEncoder:
        super().train(mode)
        if self.settings.projection_trainable:
            self.clear_cache()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states = self.backbone.encode(input_ids, attention_mask)
        pooled = masked_mean_pool(hidden_states, attention_mask)
        pooled = pooled.to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        embedding = self.projection(pooled)
        if self.settings.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def _encode_uncached(self, texts: Sequence[str]) -> Tensor:
        tokenized = self.backbone.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.backbone.runtime_device)
        attention_mask = tokenized["attention_mask"].to(self.backbone.runtime_device)
        return self(input_ids, attention_mask)

    def encode_texts(self, texts: Sequence[str], *, use_cache: bool = True) -> Tensor:
        if not texts:
            raise ValueError("texts cannot be empty")
        normalized_texts = tuple(str(text).strip() for text in texts)
        if any(not text for text in normalized_texts):
            raise ValueError("task descriptions cannot be empty")

        if not use_cache or not self.cache_enabled:
            return self._encode_uncached(normalized_texts)

        missing = tuple(dict.fromkeys(text for text in normalized_texts if text not in self._cache))
        if missing:
            with torch.no_grad():
                encoded = self._encode_uncached(missing)
            for text, embedding in zip(missing, encoded, strict=True):
                self._cache[text] = embedding.detach().cpu()
        return torch.stack(
            [self._cache[text] for text in normalized_texts],
            dim=0,
        ).to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )

    def encode_descriptions(
        self,
        descriptions: Sequence[TaskDescription],
        *,
        use_cache: bool = True,
    ) -> Tensor:
        return self.encode_texts(
            [description.serialize() for description in descriptions],
            use_cache=use_cache,
        )

