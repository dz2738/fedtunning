"""Cold-start experiment with description-visible, update-hidden clients."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from data.schema import DataSplit
from trainer.simulator import FederatedSimulator, SimulationConfig

try:
    from scripts.train import ExperimentRuntime, build_runtime
except ImportError:
    from train import ExperimentRuntime, build_runtime


LOGGER = logging.getLogger(__name__)


def select_holdout_clients(
    runtime: ExperimentRuntime,
    *,
    seed: int,
) -> tuple[str, ...]:
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[str]] = {}
    for client_id, client in runtime.clients.items():
        by_task.setdefault(client.state.task_id, []).append(client_id)
    return tuple(
        sorted(str(rng.choice(sorted(client_ids))))
        for _, client_ids in sorted(by_task.items())
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = build_runtime(config)
    if not bool(config.experiment.holdout.expose_task_description):
        raise ValueError("current semantic cold-start protocol requires task descriptions")
    holdout_ids = select_holdout_clients(runtime, seed=int(config.seed))
    training_clients = {
        client_id: client
        for client_id, client in runtime.clients.items()
        if client_id not in holdout_ids
    }
    runtime.server.initialize_groups(
        {client_id: client.state for client_id, client in runtime.clients.items()}
    )
    simulator = FederatedSimulator(
        server=runtime.server,
        clients=training_clients,
        config=SimulationConfig.from_mapping(config.experiment),
        seed=int(config.seed),
        evaluator=None,
    )
    simulator.run()
    evaluations = [
        runtime.evaluator.evaluate(
            server=runtime.server,
            clients=runtime.clients,
            client_ids=holdout_ids,
            adaptation_steps=int(steps),
            seed=int(config.seed),
            split=DataSplit.TEST,
        )
        for steps in config.experiment.adaptation_steps
    ]
    output = {
        "protocol": "transductive_semantic_descriptions_only",
        "holdout_clients": holdout_ids,
        "training_clients": tuple(sorted(training_clients)),
        "evaluations": [asdict(value) for value in evaluations],
    }
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cold_start.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("evaluated %d cold-start clients", len(holdout_ids))


if __name__ == "__main__":
    main()
