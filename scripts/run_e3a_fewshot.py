"""E3-A: transductive cold-start H-curve over coordinate initializers.

Hold out one client per task. Task descriptions of holdout clients remain
visible during grouping, but their data and updates are not used in training.
This is not an inductive unseen-task protocol; use ``run_e3b_holdout_task.py``
for whole-task holdout.
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
    evaluate_initializer_curve,
    select_one_client_per_task,
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
    if not bool(holdout.get("expose_task_description", True)):
        raise ValueError("E3-A requires visible task descriptions")
    initializers = tuple(experiment.get("initializers", COLD_START_INITIALIZERS))
    unknown = [name for name in initializers if name not in COLD_START_INITIALIZERS]
    if unknown:
        raise ValueError(f"unknown E3-A initializers: {unknown}")
    adaptation_steps = tuple(int(step) for step in experiment.get("adaptation_steps", (0, 1, 3, 5, 10)))

    runtime = build_runtime(config)
    holdout_ids = select_one_client_per_task(runtime.clients, seed=int(config.seed))
    training_clients = {
        client_id: client
        for client_id, client in runtime.clients.items()
        if client_id not in set(holdout_ids)
    }
    runtime.server.initialize_groups(
        {client_id: client.state for client_id, client in runtime.clients.items()}
    )
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True),
        encoding="utf-8",
    )
    (output_dir / "grouping_report.json").write_text(
        json.dumps(runtime.server.grouping_report(), ensure_ascii=False, indent=2),
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
        "protocol": "transductive_semantic_descriptions_only",
        "holdout_strategy": holdout.get("strategy", "one_client_per_task"),
        "holdout_clients": holdout_ids,
        "training_clients": tuple(sorted(training_clients)),
        "adaptation_steps": adaptation_steps,
        "initializers": initializers,
        "by_initializer": by_initializer,
    }
    (output_dir / "e3a_fewshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "E3-A holdout_clients=%d initializers=%d steps=%s",
        len(holdout_ids),
        len(initializers),
        list(adaptation_steps),
    )


if __name__ == "__main__":
    main()
