"""Build pairwise empirical transfer and semantic-similarity matrices."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from baselines.local_prompt import (
    ClientPromptObjective,
    PromptOptimizationConfig,
    optimize_prompt,
)
from client.client import FederatedClient
from model.soft_prompt import initialize_prompt_

try:
    from scripts.train import build_runtime
except ImportError:
    from train import build_runtime


LOGGER = logging.getLogger(__name__)


def _objective(client: FederatedClient) -> ClientPromptObjective:
    support = client._tokenize_batches(
        client.data.support,
        batch_size=8,
        shuffle=False,
        seed=0,
    )
    test = client._tokenize_batches(
        client.data.test,
        batch_size=8,
        shuffle=False,
        seed=0,
    )

    def train_loss(prompt: Tensor, step: int) -> Tensor:
        return client.backbone(support[step % len(support)], prompt_embeddings=prompt).loss

    def eval_loss(prompt: Tensor) -> Tensor:
        losses = []
        counts = []
        for batch in test:
            count = int(batch.input_ids.shape[0])
            losses.append(client.backbone(batch, prompt_embeddings=prompt).loss * count)
            counts.append(count)
        return torch.stack(losses).sum() / sum(counts)

    return ClientPromptObjective(
        client_id=client.state.client_id,
        train_loss=train_loss,
        eval_loss=eval_loss,
        num_examples=len(client.data.support),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = build_runtime(config)
    client_ids = tuple(sorted(runtime.clients))
    prompt = torch.empty(
        int(config.model.prompt.length),
        runtime.backbone.hidden_size,
        device=runtime.backbone.runtime_device,
        dtype=runtime.backbone.compute_dtype,
    )
    initialize_prompt_(
        prompt,
        strategy=str(config.model.prompt.init),
        init_std=float(config.model.prompt.init_std),
        token_embedding_weight=runtime.backbone.input_embeddings.weight,
    )
    optimization = PromptOptimizationConfig(
        steps=int(config.method.inner_loop.steps),
        learning_rate=float(config.method.inner_loop.residual_lr),
        weight_decay=float(config.method.inner_loop.residual_weight_decay),
    )
    objectives = {
        client_id: _objective(runtime.clients[client_id]) for client_id in client_ids
    }
    local_updates = {
        client_id: optimize_prompt(prompt, objectives[client_id], optimization)
        for client_id in client_ids
    }
    local_losses = {
        client_id: update.eval_loss for client_id, update in local_updates.items()
    }
    transfer = np.empty((len(client_ids), len(client_ids)), dtype=np.float64)
    for source_index, source_id in enumerate(client_ids):
        source_prompt = local_updates[source_id].prompt
        for target_index, target_id in enumerate(client_ids):
            target_loss = objectives[target_id].eval_loss(source_prompt)
            transfer[source_index, target_index] = (
                local_losses[target_id] - float(target_loss.detach())
            )

    embeddings = runtime.task_encoder.encode_descriptions(
        [runtime.clients[client_id].state.description for client_id in client_ids]
    ).detach().float()
    semantic = (embeddings @ embeddings.transpose(0, 1)).cpu().numpy()
    output = {
        "client_ids": client_ids,
        "gain_definition": "target_local_test_loss - source_prompt_target_test_loss",
        "transfer": transfer.tolist(),
        "semantic_cosine": semantic.tolist(),
    }
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transfer_matrix.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("built %d x %d transfer matrix", len(client_ids), len(client_ids))


if __name__ == "__main__":
    main()
