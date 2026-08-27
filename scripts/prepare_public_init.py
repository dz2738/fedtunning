"""Learn and save the public-proxy FedTaskPrompt initialization artifact."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from data.federated_data import build_federated_data
from data.schema import TextExample
from model.backbone import BackboneBatch, SharedBackbone, SharedBackboneConfig
from model.coordinate_generator import CoordinateGenerator, CoordinateGeneratorConfig
from model.task_encoder import TaskEncoderConfig, TaskSemanticEncoder
from scripts.train import plain_mapping, set_reproducibility
from server.public_initialization import (
    PublicInitializationConfig,
    PublicTaskObjective,
    build_public_initialization,
    save_public_initialization,
)

LOGGER = logging.getLogger(__name__)


def _batched(values: Sequence[TextExample], batch_size: int) -> tuple[Sequence[TextExample], ...]:
    return tuple(values[start : start + batch_size] for start in range(0, len(values), batch_size))


def _task_loss(
    backbone: SharedBackbone,
    batches: Sequence[BackboneBatch],
) -> Callable[[Tensor], Tensor]:
    example_counts = tuple(int(batch.input_ids.shape[0]) for batch in batches)
    total_examples = sum(example_counts)

    def loss(prompt: Tensor) -> Tensor:
        values = [
            backbone(batch, prompt_embeddings=prompt).loss * count
            for batch, count in zip(batches, example_counts, strict=True)
        ]
        return torch.stack(values).sum() / total_examples

    return loss


def _task_accumulate_backward(
    backbone: SharedBackbone,
    batches: Sequence[BackboneBatch],
) -> Callable[[Tensor, Tensor], Tensor]:
    example_counts = tuple(int(batch.input_ids.shape[0]) for batch in batches)
    total_examples = sum(example_counts)

    def accumulate(prompt: Tensor, scale: Tensor) -> Tensor:
        total = prompt.new_zeros(())
        scale = scale.to(device=prompt.device, dtype=prompt.dtype)
        for batch, count in zip(batches, example_counts, strict=True):
            value = backbone(batch, prompt_embeddings=prompt).loss * (
                count / total_examples
            )
            (scale * value).backward()
            total = total + value.detach()
        return scale.detach() * total

    return accumulate


def prepare_public_initialization(config: DictConfig) -> Path:
    seed = int(config.seed)
    set_reproducibility(seed, deterministic=bool(config.deterministic))
    public_config = PublicInitializationConfig.from_mapping(
        plain_mapping(config.method.public_initialization)
    )
    if not public_config.checkpoint_path:
        raise ValueError("method.public_initialization.checkpoint_path must be configured")

    data = build_federated_data(config.public_data, seed=seed + 1_000_000)
    backbone = SharedBackbone(
        SharedBackboneConfig.from_mapping(plain_mapping(config.model)),
        device=str(config.device),
    )
    task_encoder = TaskSemanticEncoder(
        backbone,
        TaskEncoderConfig.from_mapping(plain_mapping(config.model.task_encoder)),
    )
    num_basis = int(config.method.num_basis)
    generator = CoordinateGenerator(
        CoordinateGeneratorConfig.from_mapping(
            plain_mapping(config.method.coordinate_generator),
            embedding_dim=task_encoder.embedding_dim,
            num_basis=num_basis,
        )
    ).to(backbone.runtime_device)

    prompt_config = plain_mapping(config.model.prompt)
    text_config = plain_mapping(config.public_data.text_template)
    public_tasks: list[PublicTaskObjective] = []
    for task_id in sorted(data.tasks):
        task_clients = data.clients_for_task(task_id)
        if not task_clients:
            raise ValueError(f"public task {task_id!r} has no examples")
        # Client views share this immutable task-level collection. The offline
        # initializer reads it once and never uses support/query/test membership.
        examples = tuple(task_clients[0].examples[: public_config.max_examples_per_task])
        tokenized = tuple(
            backbone.tokenize(
                [example.input_text for example in batch],
                [example.target_text for example in batch],
                max_source_length=int(text_config.get("max_source_length", 384)),
                max_target_length=int(text_config.get("max_target_length", 96)),
            )
            for batch in _batched(examples, public_config.batch_size)
        )
        embedding = task_encoder.encode_descriptions([data.tasks[task_id].description])[0].detach()
        public_tasks.append(
            PublicTaskObjective(
                task_id=task_id,
                embedding=embedding,
                num_examples=len(examples),
                loss=_task_loss(backbone, tokenized),
                accumulate_backward=_task_accumulate_backward(backbone, tokenized),
            )
        )

    initial_center = torch.zeros(
        int(prompt_config["length"]),
        backbone.hidden_size,
        device=backbone.runtime_device,
        dtype=torch.float32,
    )
    artifact = build_public_initialization(
        public_tasks,
        generator,
        initial_center,
        num_basis=num_basis,
        config=public_config,
    )
    destination = save_public_initialization(public_config.checkpoint_path, artifact)
    public_manifest = {
        "seed": seed + 1_000_000,
        "public_data_config": OmegaConf.to_container(config.public_data, resolve=True),
        "tasks": {
            task_id: {
                "num_examples": len(data.clients_for_task(task_id)[0].examples),
                "example_ids": [
                    example.example_id
                    for example in data.clients_for_task(task_id)[0].examples
                ],
            }
            for task_id in sorted(data.tasks)
        },
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "saved public initialization path=%s tasks=%d basis=%d",
        destination,
        len(public_tasks),
        num_basis,
    )
    return destination


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(config.logging.level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    prepare_public_initialization(config)


if __name__ == "__main__":
    main()
