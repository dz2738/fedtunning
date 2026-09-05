"""E3-B: inductive whole-task holdout.

Train without every client of one task, then admit that task from its
description and evaluate the same initializer x H-step grid as E3-A.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from data.schema import DataSplit
from trainer.holdout import (
    COLD_START_INITIALIZERS,
    clients_for_task,
    evaluate_initializer_curve,
)
from trainer.simulator import FederatedSimulator, SimulationConfig

try:
    from scripts.train import build_runtime, plain_mapping
except ImportError:
    from train import build_runtime, plain_mapping

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    experiment = plain_mapping(config.experiment)
    holdout = dict(experiment.get("holdout", {}))
    task_id = str(holdout.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("E3-B requires experiment.holdout.task_id")
    initializers = tuple(experiment.get("initializers", COLD_START_INITIALIZERS))
    unknown = [name for name in initializers if name not in COLD_START_INITIALIZERS]
    if unknown:
        raise ValueError(f"unknown E3-B initializers: {unknown}")
    adaptation_steps = tuple(
        int(step) for step in experiment.get("adaptation_steps", (0, 1, 3, 5, 10))
    )

    runtime = build_runtime(config)
    holdout_ids = clients_for_task(runtime.clients, task_id)
    training_clients = {
        client_id: client
        for client_id, client in runtime.clients.items()
        if client_id not in set(holdout_ids)
    }
    if not training_clients:
        raise ValueError("E3-B holdout leaves no training clients")
    runtime.server.initialize_groups(
        {client_id: client.state for client_id, client in training_clients.items()}
    )
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )
    simulator = FederatedSimulator(
        server=runtime.server,
        clients=training_clients,
        config=SimulationConfig.from_mapping(experiment),
        seed=int(config.seed),
        evaluator=None,
    )
    simulator.run()
    admitted_group = runtime.server.admit_task(
        task_id=task_id,
        clients={client_id: client.state for client_id, client in runtime.clients.items()},
    )
    (output_dir / "grouping_report.json").write_text(
        json.dumps(runtime.server.grouping_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    by_initializer = {
        initializer: [
            asdict(summary)
            for summary in evaluate_initializer_curve(
                server=runtime.server,
                evaluator=runtime.evaluator,
                clients=runtime.clients,
                holdout_ids=holdout_ids,
                training_client_ids=tuple(sorted(training_clients)),
                initializer=initializer,
                adaptation_steps=adaptation_steps,
                seed=int(config.seed),
                split=DataSplit.TEST,
            )
        ]
        for initializer in initializers
    }
    payload = {
        "protocol": "inductive_holdout_entire_task",
        "holdout_task": task_id,
        "admitted_group": admitted_group,
        "holdout_clients": holdout_ids,
        "training_clients": tuple(sorted(training_clients)),
        "adaptation_steps": adaptation_steps,
        "initializers": initializers,
        "by_initializer": by_initializer,
    }
    (output_dir / "e3b_holdout_task.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "E3-B task=%s admitted_group=%s holdout_clients=%d",
        task_id,
        admitted_group,
        len(holdout_ids),
    )


if __name__ == "__main__":
    main()
