"""Main FedTaskPrompt training entry point."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from client.client import FederatedClient, TextBatchConfig
from client.state import ClientState, InnerLoopConfig
from data.federated_data import FederatedData, build_federated_data
from data.schema import DataSplit
from model.backbone import SharedBackbone, SharedBackboneConfig
from model.coordinate_generator import CoordinateGenerator, CoordinateGeneratorConfig
from model.task_encoder import TaskEncoderConfig, TaskSemanticEncoder
from server.aggregation import ServerOptimizerConfig
from server.basis_maintenance import BasisMaintenanceConfig
from server.grouping import GroupingConfig
from server.public_initialization import (
    PublicInitializationArtifact,
    PublicInitializationConfig,
    load_public_initialization,
)
from server.server import FedTaskPromptServer
from trainer.checkpoint import load_checkpoint, save_checkpoint
from trainer.evaluator import FederatedEvaluator
from trainer.simulator import FederatedSimulator, SimulationConfig, SimulationResult

LOGGER = logging.getLogger(__name__)
RESULT_FILES = (
    "run_metadata.json",
    "resolved_config.yaml",
    "rounds.jsonl",
    "client_adaptation.jsonl",
    "evaluations.jsonl",
    "test_evaluation.json",
    "best_validation.json",
    "grouping_report.json",
)


@dataclass(slots=True)
class ExperimentRuntime:
    data: FederatedData
    backbone: SharedBackbone
    task_encoder: TaskSemanticEncoder
    clients: dict[str, FederatedClient]
    server: FedTaskPromptServer
    evaluator: FederatedEvaluator
    simulator: FederatedSimulator
    public_initialization: Mapping[str, Any] | None


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


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_metadata(config: DictConfig) -> dict[str, Any]:
    packages = {}
    for package in ("torch", "transformers", "datasets", "hydra-core"):
        try:
            packages[package] = version(package)
        except Exception:
            packages[package] = None
    return {
        "run_id": str(config.run_id),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": int(config.seed),
    }


def _prepare_output_dir(
    output_dir: Path,
    *,
    resume_from: str | Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicts = [
        output_dir / name
        for name in RESULT_FILES
        if (output_dir / name).exists()
    ]
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        conflicts.append(checkpoint_dir)
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        if conflicts and resume_path.parent != checkpoint_dir.resolve():
            raise ValueError(
                "an existing result directory can only be resumed from its own "
                f"checkpoints directory: {checkpoint_dir}"
            )
        return
    if conflicts:
        formatted = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            "refusing to append a new run to an existing result directory; "
            f"choose a new run_id/output_dir. Existing paths: {formatted}"
        )


def _load_public_artifact(
    method_config: Mapping[str, Any],
    *,
    prompt_length: int,
    hidden_size: int,
    num_basis: int,
    embedding_dim: int,
) -> tuple[PublicInitializationArtifact | None, Mapping[str, Any] | None]:
    settings = PublicInitializationConfig.from_mapping(
        plain_mapping(method_config.get("public_initialization", {}))
    )
    if not settings.enabled:
        return None, None
    if not settings.checkpoint_path:
        if settings.require_checkpoint:
            raise ValueError("required public initialization has no checkpoint_path")
        LOGGER.warning("public initialization enabled without checkpoint_path; using random state")
        return None, None
    path = Path(settings.checkpoint_path).expanduser().resolve()
    if not path.is_file():
        if settings.require_checkpoint:
            raise FileNotFoundError(f"public initialization checkpoint not found: {path}")
        LOGGER.warning("public initialization checkpoint not found at %s; using random state", path)
        return None, None
    artifact = load_public_initialization(path)
    artifact.validate(
        prompt_length=prompt_length,
        hidden_size=hidden_size,
        num_basis=num_basis,
        embedding_dim=embedding_dim,
    )
    metadata = {
        "path": str(path),
        "task_ids": list(artifact.task_ids),
        "num_basis": int(artifact.basis.shape[0]),
        "singular_values": artifact.singular_values.tolist(),
    }
    return artifact, metadata


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
    inner_loop = InnerLoopConfig.from_mapping(plain_mapping(config.method.inner_loop))
    prompt_config = plain_mapping(config.model.prompt)
    public_artifact, public_metadata = _load_public_artifact(
        method_config,
        prompt_length=int(prompt_config["length"]),
        hidden_size=backbone.hidden_size,
        num_basis=num_basis,
        embedding_dim=task_encoder.embedding_dim,
    )
    if public_artifact is not None:
        generator.load_state_dict(public_artifact.generator_state, strict=True)
    server = FedTaskPromptServer(
        task_encoder=task_encoder,
        coordinate_generator=generator,
        grouping=GroupingConfig.from_mapping(plain_mapping(config.method.grouping)),
        optimizer=ServerOptimizerConfig.from_mapping(plain_mapping(config.method.server_optimizer)),
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
        public_center=None if public_artifact is None else public_artifact.center,
        public_basis=None if public_artifact is None else public_artifact.basis,
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
                description=client_data.description,
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
        public_initialization=public_metadata,
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _checkpoint_callback(
    runtime: ExperimentRuntime,
    output_dir: Path,
    checkpoint_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    tensorboard: SummaryWriter,
) -> Callable[[Any], None]:
    interval = int(checkpoint_config.get("save_every_rounds", 10))
    keep_last = int(checkpoint_config.get("keep_last", 2))
    if interval <= 0 or keep_last <= 0:
        raise ValueError("checkpoint interval and keep_last must be positive")

    def callback(summary: Any) -> None:
        _append_jsonl(output_dir / "rounds.jsonl", asdict(summary))
        _append_jsonl(
            output_dir / "client_adaptation.jsonl",
            {
                "round_number": summary.round_number,
                "clients": [asdict(trace) for trace in summary.client_adaptation],
            },
        )
        steps_per_round = runtime.server.inner_loop.steps
        for trace in summary.client_adaptation:
            client_tag = trace.client_id.replace("/", "__")
            for step, loss in enumerate(trace.training_support_losses, start=1):
                global_step = (summary.round_number - 1) * steps_per_round + step
                tensorboard.add_scalar(
                    f"client_finetune/train_batch_loss/{client_tag}",
                    loss,
                    global_step,
                )
            for step, loss in enumerate(trace.support_monitor_losses):
                global_step = (
                    (summary.round_number - 1) * (steps_per_round + 1) + step
                )
                tensorboard.add_scalar(
                    f"client_finetune/fixed_support_loss/{client_tag}",
                    loss,
                    global_step,
                )
            tensorboard.add_scalar(
                f"client_finetune/query_loss/{client_tag}",
                trace.query_loss,
                summary.round_number,
            )
        tensorboard.flush()
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
            extra={
                "round_number": summary.round_number,
                "public_initialization": runtime.public_initialization,
                "run_metadata": dict(run_metadata),
            },
        )
        checkpoints = sorted(destination.parent.glob("round_*.pt"))
        for obsolete in checkpoints[:-keep_last]:
            obsolete.unlink()

    return callback


def run_training(config: DictConfig) -> tuple[ExperimentRuntime, SimulationResult]:
    output_dir = Path(str(config.output_dir)).resolve()
    checkpoint_config = plain_mapping(config.checkpoint)
    resume_from = checkpoint_config.get("resume_from")
    _prepare_output_dir(output_dir, resume_from=resume_from)
    metadata = _run_metadata(config)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )
    runtime = build_runtime(config)
    if resume_from:
        load_checkpoint(
            Path(str(resume_from)).expanduser().resolve(),
            server=runtime.server,
            clients=runtime.clients,
            map_location=runtime.backbone.runtime_device,
        )
    else:
        runtime.simulator.initialize()
    (output_dir / "grouping_report.json").write_text(
        json.dumps(
            runtime.server.grouping_report(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    evaluation_steps = runtime.simulator.config.eval_inner_steps
    selection_steps = int(
        config.experiment.get(
            "selection_adaptation_steps",
            max(evaluation_steps),
        )
    )
    if selection_steps not in set(evaluation_steps):
        raise ValueError(
            "experiment.selection_adaptation_steps must be included in eval_inner_steps"
        )
    best_path = output_dir / "checkpoints" / "best.pt"
    best_record_path = output_dir / "best_validation.json"
    best_loss = float("inf")
    if resume_from and best_record_path.is_file():
        best_loss = float(json.loads(best_record_path.read_text())["mean_validation_loss"])

    def on_evaluation(summary: Any) -> None:
        nonlocal best_loss
        _append_jsonl(output_dir / "evaluations.jsonl", asdict(summary))
        LOGGER.info(
            "evaluation split=%s round=%d steps=%d loss=%.6f",
            summary.split,
            summary.round_number,
            summary.adaptation_steps,
            summary.mean_test_loss,
        )
        if (
            summary.split is DataSplit.VALIDATION
            and summary.adaptation_steps == selection_steps
            and summary.mean_test_loss < best_loss
        ):
            best_loss = summary.mean_test_loss
            best_record = {
                "round_number": summary.round_number,
                "adaptation_steps": summary.adaptation_steps,
                "mean_validation_loss": summary.mean_test_loss,
            }
            save_checkpoint(
                best_path,
                server=runtime.server,
                clients=runtime.clients,
                extra={
                    **best_record,
                    "public_initialization": runtime.public_initialization,
                    "run_metadata": metadata,
                },
            )
            best_record_path.write_text(
                json.dumps(best_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    tensorboard = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        result = runtime.simulator.run(
            on_round=_checkpoint_callback(
                runtime,
                output_dir,
                checkpoint_config,
                metadata,
                tensorboard,
            ),
            on_evaluation=on_evaluation,
        )
    finally:
        tensorboard.close()
    final_path = output_dir / "checkpoints" / "final.pt"
    save_checkpoint(
        final_path,
        server=runtime.server,
        clients=runtime.clients,
        extra={
            "completed": True,
            "public_initialization": runtime.public_initialization,
            "run_metadata": metadata,
        },
    )
    selected_path = best_path if best_path.is_file() else final_path
    load_checkpoint(
        selected_path,
        server=runtime.server,
        clients=runtime.clients,
        map_location=runtime.backbone.runtime_device,
    )
    test_summaries = [
        runtime.evaluator.evaluate(
            server=runtime.server,
            clients=runtime.clients,
            adaptation_steps=int(steps),
            seed=int(config.seed),
            split=DataSplit.TEST,
        )
        for steps in evaluation_steps
    ]
    (output_dir / "test_evaluation.json").write_text(
        json.dumps(
            {
                "checkpoint": str(selected_path),
                "summaries": [asdict(summary) for summary in test_summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if selected_path != final_path:
        load_checkpoint(
            final_path,
            server=runtime.server,
            clients=runtime.clients,
            map_location=runtime.backbone.runtime_device,
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
