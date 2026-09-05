"""Train one prompt baseline on the shared frozen backbone."""

from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from trainer.baseline_loop import BASELINE_METHODS, run_baseline_training

try:
    from scripts.train import build_runtime, plain_mapping
except ImportError:
    from train import build_runtime, plain_mapping

LOGGER = logging.getLogger(__name__)


def _baseline_name(config: DictConfig) -> str:
    experiment = plain_mapping(config.experiment)
    name = str(experiment.get("baseline_method", experiment.get("method", "")))
    if name not in BASELINE_METHODS:
        raise ValueError(
            "run_baseline.py requires experiment.baseline_method in "
            f"{BASELINE_METHODS}, got {name!r}"
        )
    return name


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(config.logging.level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    method_name = _baseline_name(config)
    runtime = build_runtime(config)
    payload: dict[str, Any] = run_baseline_training(
        config,
        runtime,
        method_name=method_name,
    )
    LOGGER.info(
        "finished baseline=%s test_summaries=%d",
        method_name,
        len(payload["summaries"]),
    )


if __name__ == "__main__":
    main()
