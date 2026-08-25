"""One frozen seq2seq backbone shared by every simulated client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model.soft_prompt import SoftPromptInjector


_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        return _DTYPE_BY_NAME[str(value).lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {value!r}") from error


def resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class SharedBackboneConfig:
    pretrained_name_or_path: str
    revision: str = "main"
    trust_remote_code: bool = False
    local_files_only: bool = False
    freeze_backbone: bool = True
    dtype: str = "bfloat16"
    attention_implementation: str | None = "sdpa"
    gradient_checkpointing: bool = False
    use_cache: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SharedBackboneConfig:
        return cls(
            pretrained_name_or_path=str(value["pretrained_name_or_path"]),
            revision=str(value.get("revision", "main")),
            trust_remote_code=bool(value.get("trust_remote_code", False)),
            local_files_only=bool(value.get("local_files_only", False)),
            freeze_backbone=bool(value.get("freeze_backbone", True)),
            dtype=str(value.get("dtype", "bfloat16")),
            attention_implementation=value.get("attention_implementation"),
            gradient_checkpointing=bool(value.get("gradient_checkpointing", False)),
            use_cache=bool(value.get("use_cache", False)),
        )


@dataclass(frozen=True, slots=True)
class BackboneBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor | None = None

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> BackboneBatch:
        return BackboneBatch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
            labels=None
            if self.labels is None
            else self.labels.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class ParameterSummary:
    total: int
    trainable: int


class SharedBackbone(nn.Module):
    """Frozen Hugging Face model reused sequentially by all clients."""

    def __init__(
        self,
        config: SharedBackboneConfig,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        super().__init__()
        self.runtime_device = resolve_device(device)
        requested_dtype = resolve_dtype(config.dtype)
        if self.runtime_device.type == "cpu" and requested_dtype is torch.float16:
            requested_dtype = torch.float32
        self.compute_dtype = requested_dtype
        self.settings = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_name_or_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must define either pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "revision": config.revision,
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
            "torch_dtype": requested_dtype,
        }
        if config.attention_implementation:
            model_kwargs["attn_implementation"] = config.attention_implementation
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.pretrained_name_or_path,
            **model_kwargs,
        )
        self.model.config.use_cache = config.use_cache
        self.model.to(self.runtime_device)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if config.freeze_backbone:
            self.freeze_parameters()
        self.model.eval()
        self.prompt_injector = SoftPromptInjector(self.model.get_input_embeddings())

    @property
    def hidden_size(self) -> int:
        embeddings = self.model.get_input_embeddings()
        return int(embeddings.embedding_dim)

    @property
    def input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def freeze_parameters(self) -> None:
        self.model.requires_grad_(False)

    def parameter_summary(self) -> ParameterSummary:
        parameters = tuple(self.model.parameters())
        return ParameterSummary(
            total=sum(parameter.numel() for parameter in parameters),
            trainable=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        )

    def train(self, mode: bool = True) -> SharedBackbone:
        # Keep dropout disabled in the frozen model so client order does not
        # introduce avoidable stochasticity. Prompt gradients still propagate.
        super().train(mode)
        self.model.eval()
        return self

    def tokenize(
        self,
        input_texts: Sequence[str],
        target_texts: Sequence[str] | None = None,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> BackboneBatch:
        if not input_texts:
            raise ValueError("input_texts cannot be empty")
        if target_texts is not None and len(input_texts) != len(target_texts):
            raise ValueError("input_texts and target_texts must have equal lengths")
        encoded = self.tokenizer(
            list(input_texts),
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        labels: Tensor | None = None
        if target_texts is not None:
            encoded_targets = self.tokenizer(
                text_target=list(target_texts),
                padding=True,
                truncation=True,
                max_length=max_target_length,
                return_tensors="pt",
            )
            labels = encoded_targets["input_ids"]
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        return BackboneBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
        )

    def forward(
        self,
        batch: BackboneBatch,
        *,
        prompt_embeddings: Tensor | None = None,
        **model_kwargs: Any,
    ) -> Any:
        batch = batch.to(self.runtime_device)
        common_kwargs = {
            "attention_mask": batch.attention_mask,
            "labels": batch.labels,
            "use_cache": self.settings.use_cache,
            "return_dict": True,
            **model_kwargs,
        }
        if prompt_embeddings is None:
            return self.model(input_ids=batch.input_ids, **common_kwargs)
        prompted = self.prompt_injector(
            batch.input_ids,
            batch.attention_mask,
            prompt_embeddings,
        )
        common_kwargs["attention_mask"] = prompted.attention_mask
        return self.model(inputs_embeds=prompted.inputs_embeds, **common_kwargs)

    def generate(
        self,
        batch: BackboneBatch,
        *,
        prompt_embeddings: Tensor | None = None,
        **generation_kwargs: Any,
    ) -> Tensor:
        batch = batch.to(self.runtime_device)
        common_kwargs = {
            "attention_mask": batch.attention_mask,
            "use_cache": self.settings.use_cache,
            **generation_kwargs,
        }
        if prompt_embeddings is None:
            return self.model.generate(input_ids=batch.input_ids, **common_kwargs)
        prompted = self.prompt_injector(
            batch.input_ids,
            batch.attention_mask,
            prompt_embeddings,
        )
        common_kwargs["attention_mask"] = prompted.attention_mask
        return self.model.generate(inputs_embeds=prompted.inputs_embeds, **common_kwargs)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Expose frozen encoder states for the later task-semantic encoder."""

        input_ids = input_ids.to(self.runtime_device)
        attention_mask = attention_mask.to(self.runtime_device)
        outputs = self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

