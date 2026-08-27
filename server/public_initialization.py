"""Public-proxy initialization for the prompt center, basis, and generator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from model.coordinate_generator import CoordinateGenerator

PromptLoss = Callable[[Tensor], Tensor]
PUBLIC_INITIALIZATION_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PublicInitializationConfig:
    enabled: bool = False
    checkpoint_path: str | None = None
    require_checkpoint: bool = False
    center_steps: int = 200
    center_lr: float = 1.0e-2
    task_steps: int = 100
    task_lr: float = 1.0e-2
    generator_steps: int = 500
    generator_lr: float = 1.0e-3
    max_examples_per_task: int = 128
    batch_size: int = 8

    def __post_init__(self) -> None:
        if min(self.center_steps, self.task_steps, self.generator_steps) < 0:
            raise ValueError("public initialization step counts must be non-negative")
        if min(self.center_lr, self.task_lr, self.generator_lr) <= 0:
            raise ValueError("public initialization learning rates must be positive")
        if min(self.max_examples_per_task, self.batch_size) <= 0:
            raise ValueError("public task example count and batch size must be positive")
        if self.enabled and self.require_checkpoint and not self.checkpoint_path:
            raise ValueError("enabled required public initialization needs checkpoint_path")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PublicInitializationConfig:
        checkpoint = value.get("checkpoint_path")
        return cls(
            enabled=bool(value.get("enabled", False)),
            checkpoint_path=None if checkpoint is None else str(checkpoint),
            require_checkpoint=bool(value.get("require_checkpoint", False)),
            center_steps=int(value.get("center_steps", 200)),
            center_lr=float(value.get("center_lr", 1.0e-2)),
            task_steps=int(value.get("task_steps", 100)),
            task_lr=float(value.get("task_lr", 1.0e-2)),
            generator_steps=int(value.get("generator_steps", 500)),
            generator_lr=float(value.get("generator_lr", 1.0e-3)),
            max_examples_per_task=int(value.get("max_examples_per_task", 128)),
            batch_size=int(value.get("batch_size", 8)),
        )


@dataclass(frozen=True, slots=True)
class PublicTaskObjective:
    task_id: str
    embedding: Tensor
    num_examples: int
    loss: PromptLoss
    accumulate_backward: Callable[[Tensor, Tensor], Tensor] | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("public task_id must be non-empty")
        if self.embedding.ndim != 1 or not self.embedding.is_floating_point():
            raise ValueError("public task embedding must be a floating-point vector")
        if self.num_examples <= 0:
            raise ValueError("public task num_examples must be positive")

    def scaled_objective(self, prompt: Tensor, scale: Tensor) -> Tensor:
        """Return scale * loss(prompt), using micro-batch backward when provided.

        `accumulate_backward` must apply `backward()` per micro-batch and return a
        detached scalar equal to the scaled mean loss. That keeps the public-center
        objective identical while avoiding one autograd graph over every task batch.
        """

        if self.accumulate_backward is not None:
            return self.accumulate_backward(prompt, scale)
        return scale.to(device=prompt.device, dtype=prompt.dtype) * self.loss(prompt)


@dataclass(frozen=True, slots=True)
class PublicInitializationArtifact:
    center: Tensor
    basis: Tensor
    task_ids: tuple[str, ...]
    task_weights: Tensor
    task_coordinates: Tensor
    task_embeddings: Tensor
    public_prototype: Tensor
    singular_values: Tensor
    generator_state: Mapping[str, Tensor]

    def validate(
        self,
        *,
        prompt_length: int | None = None,
        hidden_size: int | None = None,
        num_basis: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        if self.center.ndim != 2 or not self.center.is_floating_point():
            raise ValueError("public center must be a floating-point prompt matrix")
        if self.basis.ndim != 3 or not self.basis.is_floating_point():
            raise ValueError("public basis must have shape [K, L_p, d]")
        if tuple(self.basis.shape[1:]) != tuple(self.center.shape):
            raise ValueError("public center and basis prompt shapes differ")
        tasks = len(self.task_ids)
        if len(set(self.task_ids)) != tasks:
            raise ValueError("public task IDs must be unique")
        if self.task_weights.shape != (tasks,):
            raise ValueError("public task weights have the wrong shape")
        if self.task_coordinates.shape != (tasks, self.basis.shape[0]):
            raise ValueError("public task coordinates have the wrong shape")
        if self.task_embeddings.ndim != 2 or self.task_embeddings.shape[0] != tasks:
            raise ValueError("public task embeddings have the wrong shape")
        if self.public_prototype.shape != (self.task_embeddings.shape[1],):
            raise ValueError("public semantic prototype has the wrong shape")
        if not torch.isclose(
            self.task_weights.sum(),
            self.task_weights.new_tensor(1.0),
            atol=1.0e-5,
        ):
            raise ValueError("public task weights must sum to one")
        if prompt_length is not None and self.center.shape[0] != prompt_length:
            raise ValueError("public center prompt length differs from the model")
        if hidden_size is not None and self.center.shape[1] != hidden_size:
            raise ValueError("public center hidden size differs from the model")
        if num_basis is not None and self.basis.shape[0] != num_basis:
            raise ValueError("public basis count differs from the method config")
        if embedding_dim is not None and self.task_embeddings.shape[1] != embedding_dim:
            raise ValueError("public embedding dimension differs from the task encoder")


def normalized_example_weights(tasks: Sequence[PublicTaskObjective]) -> Tensor:
    if not tasks:
        raise ValueError("public task collection cannot be empty")
    counts = torch.tensor([task.num_examples for task in tasks], dtype=torch.float64)
    return counts / counts.sum()


def weighted_prompt_svd(
    task_prompts: Tensor,
    center: Tensor,
    task_weights: Tensor,
    *,
    num_basis: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply Eq. (17)--(18) to weighted public-task prompt differences."""

    if task_prompts.ndim != 3 or center.ndim != 2:
        raise ValueError("task prompts and center must have shapes [M,L,d] and [L,d]")
    if tuple(task_prompts.shape[1:]) != tuple(center.shape):
        raise ValueError("task prompts and center have incompatible shapes")
    if task_weights.shape != (task_prompts.shape[0],):
        raise ValueError("task_weights must contain one value per public task")
    if num_basis <= 0:
        raise ValueError("num_basis must be positive")
    weights = task_weights.to(device=task_prompts.device, dtype=torch.float32)
    if bool((weights < 0).any()) or not torch.isfinite(weights).all():
        raise ValueError("task_weights must be finite and non-negative")
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    differences = (task_prompts - center).reshape(task_prompts.shape[0], -1).float()
    weighted_matrix = differences.transpose(0, 1) * weights.sqrt().unsqueeze(0)
    left_vectors, singular_values, _ = torch.linalg.svd(weighted_matrix, full_matrices=False)
    if singular_values.numel() == 0:
        rank = 0
    else:
        tolerance = (
            max(weighted_matrix.shape)
            * torch.finfo(singular_values.dtype).eps
            * float(singular_values.max())
        )
        rank = int((singular_values > tolerance).sum())
    if rank < num_basis:
        raise ValueError(
            "public prompt differences have insufficient rank: "
            f"rank={rank}, requested num_basis={num_basis}; add diverse public tasks "
            "or reduce method.num_basis"
        )
    directions = left_vectors[:, :num_basis]
    coordinates = differences @ directions
    basis = directions.transpose(0, 1).reshape(
        num_basis,
        center.shape[0],
        center.shape[1],
    )
    return basis.to(task_prompts.dtype), coordinates.to(task_prompts.dtype), singular_values


def _optimize_prompt(
    initial_prompt: Tensor,
    objective: Callable[[Tensor], Tensor],
    *,
    steps: int,
    learning_rate: float,
) -> Tensor:
    prompt = initial_prompt.detach().float().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([prompt], lr=learning_rate, weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(prompt)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("public prompt objective must return one finite scalar")
        if loss.requires_grad:
            loss.backward()
        elif prompt.grad is None:
            raise ValueError("public prompt objective produced no gradients")
        optimizer.step()
    return prompt.detach()


def learn_public_center(
    tasks: Sequence[PublicTaskObjective],
    initial_center: Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> Tensor:
    weights = normalized_example_weights(tasks).to(initial_center.device)

    def joint_loss(prompt: Tensor) -> Tensor:
        parts = [
            task.scaled_objective(prompt, weight.to(dtype=prompt.dtype))
            for weight, task in zip(weights, tasks, strict=True)
        ]
        return parts[0] if len(parts) == 1 else torch.stack(parts).sum()

    return _optimize_prompt(
        initial_center,
        joint_loss,
        steps=steps,
        learning_rate=learning_rate,
    )


def adapt_public_task_prompts(
    tasks: Sequence[PublicTaskObjective],
    center: Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> Tensor:
    return torch.stack(
        [
            _optimize_prompt(
                center,
                lambda prompt, current=task: current.scaled_objective(
                    prompt,
                    prompt.new_tensor(1.0),
                ),
                steps=steps,
                learning_rate=learning_rate,
            )
            for task in tasks
        ],
        dim=0,
    )


def pretrain_coordinate_generator(
    generator: CoordinateGenerator,
    task_embeddings: Tensor,
    public_prototype: Tensor,
    target_coordinates: Tensor,
    task_weights: Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> None:
    parameter = next(generator.parameters())
    embeddings = task_embeddings.to(device=parameter.device, dtype=parameter.dtype)
    prototype = public_prototype.to(device=parameter.device, dtype=parameter.dtype)
    targets = target_coordinates.to(device=parameter.device, dtype=parameter.dtype)
    weights = task_weights.to(device=parameter.device, dtype=parameter.dtype)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate, weight_decay=0.0)
    generator.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = generator(embeddings, prototype.expand_as(embeddings))
        per_task = (predictions - targets).square().sum(dim=-1)
        loss = (weights * per_task).sum()
        loss.backward()
        optimizer.step()


def build_public_initialization(
    tasks: Sequence[PublicTaskObjective],
    generator: CoordinateGenerator,
    initial_center: Tensor,
    *,
    num_basis: int,
    config: PublicInitializationConfig,
) -> PublicInitializationArtifact:
    weights = normalized_example_weights(tasks).float()
    center = learn_public_center(
        tasks,
        initial_center,
        steps=config.center_steps,
        learning_rate=config.center_lr,
    )
    task_prompts = adapt_public_task_prompts(
        tasks,
        center,
        steps=config.task_steps,
        learning_rate=config.task_lr,
    )
    basis, coordinates, singular_values = weighted_prompt_svd(
        task_prompts,
        center,
        weights,
        num_basis=num_basis,
    )
    embeddings = torch.stack([task.embedding.detach().float().cpu() for task in tasks])
    prototype = F.normalize((weights[:, None] * embeddings).sum(dim=0), p=2, dim=0)
    pretrain_coordinate_generator(
        generator,
        embeddings,
        prototype,
        coordinates,
        weights,
        steps=config.generator_steps,
        learning_rate=config.generator_lr,
    )
    artifact = PublicInitializationArtifact(
        center=center.detach().cpu(),
        basis=basis.detach().cpu(),
        task_ids=tuple(task.task_id for task in tasks),
        task_weights=weights.detach().cpu(),
        task_coordinates=coordinates.detach().cpu(),
        task_embeddings=embeddings,
        public_prototype=prototype.detach().cpu(),
        singular_values=singular_values.detach().cpu(),
        generator_state={
            name: value.detach().cpu().clone() for name, value in generator.state_dict().items()
        },
    )
    artifact.validate(num_basis=num_basis, embedding_dim=generator.embedding_dim)
    return artifact


def save_public_initialization(
    path: str | Path,
    artifact: PublicInitializationArtifact,
) -> Path:
    artifact.validate()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": PUBLIC_INITIALIZATION_FORMAT_VERSION,
            "artifact": artifact,
        },
        destination,
    )
    return destination


def load_public_initialization(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> PublicInitializationArtifact:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    if int(payload.get("format_version", -1)) != PUBLIC_INITIALIZATION_FORMAT_VERSION:
        raise ValueError("unsupported public initialization artifact version")
    artifact = payload.get("artifact")
    if not isinstance(artifact, PublicInitializationArtifact):
        raise TypeError("public initialization file does not contain a valid artifact")
    artifact.validate()
    return artifact
