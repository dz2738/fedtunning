"""Main FedTaskPrompt training entry point."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from client.client import FederatedClient, TextBatchConfig
from client.state import ClientState, InnerLoopConfig
from data.federated_data import FederatedData, build_federated_data
from model.backbone import SharedBackbone, SharedBackboneConfig
from model.coordinate_generator import CoordinateGenerator, CoordinateGeneratorConfig
from model.task_encoder import TaskEncoderConfig, TaskSemanticEncoder
from server.aggregation import ServerOptimizerConfig
from server.basis_maintenance import BasisMaintenanceConfig
from server.grouping import GroupingConfig
from server.server import FedTaskPromptServer
from trainer.checkpoint import load_checkpoint, save_checkpoint
from trainer.evaluator import FederatedEvaluator
from trainer.simulator import FederatedSimulator, SimulationConfig, SimulationResult


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentRuntime:
    data: FederatedData
    backbone: SharedBackbone
    task_encoder: TaskSemanticEncoder
    clients: dict[str, FederatedClient]
    server: FedTaskPromptServer
    evaluator: FederatedEvaluator
    simulator: FederatedSimulator


def plain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    converted = OmegaConf.to_container(value, resolve=True)
    if not isinstance(converted, Mapping):
        raise TypeError("configuration section must resolve to a mapping")
    return converted


def set_reproducibility(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def build_runtime(config: DictConfig) -> ExperimentRuntime:
    """Construct one backbone and all lightweight server/client state."""

    seed = int(config.seed)
    set_reproducibility(seed, deterministic=bool(config.deterministic))
    data = build_federated_data(config.data, seed=seed)

    model_config = plain_mapping(config.model)
    backbone = SharedBackbone(
        SharedBackboneConfig.from_mapping(model_config),
        device=str(config.device),
    )
    task_encoder = TaskSemanticEncoder(
        backbone,
        TaskEncoderConfig.from_mapping(plain_mapping(config.model.task_encoder)),
    )
    method_config = plain_mapping(config.method)
    num_basis = int(method_config["num_basis"])
    generator = CoordinateGenerator(
        CoordinateGeneratorConfig.from_mapping(
            plain_mapping(config.method.coordinate_generator),
            embedding_dim=task_encoder.embedding_dim,
            num_basis=num_basis,
        )
    ).to(backbone.runtime_device)
    inner_loop = InnerLoopConfig.from_mapping(
        plain_mapping(config.method.inner_loop)
    )
    prompt_config = plain_mapping(config.model.prompt)
    server = FedTaskPromptServer(
        task_encoder=task_encoder,
        coordinate_generator=generator,
        grouping=GroupingConfig.from_mapping(
            plain_mapping(config.method.grouping)
        ),
        optimizer=ServerOptimizerConfig.from_mapping(
            plain_mapping(config.method.server_optimizer)
        ),
        inner_loop=inner_loop,
        basis_maintenance=BasisMaintenanceConfig.from_mapping(
            plain_mapping(config.method.basis_maintenance)
        ),
        prompt_length=int(prompt_config["length"]),
        hidden_size=backbone.hidden_size,
        num_basis=num_basis,
        prompt_init_std=float(prompt_config.get("init_std", 0.02)),
        projection_eps=float(method_config.get("projection_eps", 1.0e-6)),
        prompt_dtype=backbone.compute_dtype,
    )
    text_config = plain_mapping(config.data.text_template)
    text_batch = TextBatchConfig(
        max_source_length=int(text_config.get("max_source_length", 384)),
        max_target_length=int(text_config.get("max_target_length", 96)),
    )
    clients = {
        client_id: FederatedClient(
            state=ClientState(
                client_id=client_id,
                task_id=client_data.task.task_id,
                description=client_data.task.description,
            ),
            data=client_data,
            backbone=backbone,
            text_batch=text_batch,
        )
        for client_id, client_data in data.clients.items()
    }
    report = plain_mapping(config.experiment.get("report", {}))
    evaluator = FederatedEvaluator(
        worst_client_fraction=float(report.get("worst_client_fraction", 0.1))
    )
    simulator = FederatedSimulator(
        server=server,
        clients=clients,
        config=SimulationConfig.from_mapping(plain_mapping(config.experiment)),
        seed=seed,
        evaluator=evaluator,
    )
    return ExperimentRuntime(
        data=data,
        backbone=backbone,
        task_encoder=task_encoder,
        clients=clients,
        server=server,
        evaluator=evaluator,
        simulator=simulator,
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _checkpoint_callback(
    runtime: ExperimentRuntime,
    output_dir: Path,
    checkpoint_config: Mapping[str, Any],
) -> Callable[[Any], None]:
    interval = int(checkpoint_config.get("save_every_rounds", 10))
    keep_last = int(checkpoint_config.get("keep_last", 2))
    if interval <= 0 or keep_last <= 0:
        raise ValueError("checkpoint interval and keep_last must be positive")

    def callback(summary: Any) -> None:
        _append_jsonl(output_dir / "rounds.jsonl", asdict(summary))
        LOGGER.info(
            "round=%d clients=%d query_loss=%.6f",
            summary.round_number,
            summary.aggregation.clients,
            summary.aggregation.mean_query_loss,
        )
        if summary.round_number % interval != 0:
            return
        destination = output_dir / "checkpoints" / f"round_{summary.round_number:06d}.pt"
        save_checkpoint(
            destination,
            server=runtime.server,
            clients=runtime.clients,
            extra={"round_number": summary.round_number},
        )
        checkpoints = sorted(destination.parent.glob("round_*.pt"))
        for obsolete in checkpoints[:-keep_last]:
            obsolete.unlink()

    return callback


def run_training(config: DictConfig) -> tuple[ExperimentRuntime, SimulationResult]:
    runtime = build_runtime(config)
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )
    checkpoint_config = plain_mapping(config.checkpoint)
    resume_from = checkpoint_config.get("resume_from")
    if resume_from:
        load_checkpoint(
            Path(str(resume_from)).expanduser().resolve(),
            server=runtime.server,
            clients=runtime.clients,
            map_location=runtime.backbone.runtime_device,
        )
    else:
        runtime.simulator.initialize()

    def on_evaluation(summary: Any) -> None:
        _append_jsonl(output_dir / "evaluations.jsonl", asdict(summary))
        LOGGER.info(
            "evaluation round=%d steps=%d test_loss=%.6f",
            summary.round_number,
            summary.adaptation_steps,
            summary.mean_test_loss,
        )

    result = runtime.simulator.run(
        on_round=_checkpoint_callback(runtime, output_dir, checkpoint_config),
        on_evaluation=on_evaluation,
    )
    save_checkpoint(
        output_dir / "checkpoints" / "final.pt",
        server=runtime.server,
        clients=runtime.clients,
        extra={"completed": True},
    )
    return runtime, result


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(config.logging.level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    runtime, result = run_training(config)
    LOGGER.info(
        "finished rounds=%d clients=%d groups=%d backbone_instances=1",
        len(result.rounds),
        len(runtime.clients),
        len(runtime.server.groups),
    )


if __name__ == "__main__":
    main()
