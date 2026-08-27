"""Restore a checkpoint and evaluate all clients at requested adaptation steps."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

from data.schema import DataSplit
from trainer.checkpoint import load_checkpoint

try:
    from scripts.train import build_runtime
except ImportError:
    from train import build_runtime


LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint = config.checkpoint.resume_from
    if not checkpoint:
        raise ValueError("set checkpoint.resume_from to the checkpoint being evaluated")
    runtime = build_runtime(config)
    load_checkpoint(
        Path(str(checkpoint)).expanduser().resolve(),
        server=runtime.server,
        clients=runtime.clients,
        map_location=runtime.backbone.runtime_device,
    )
    steps = tuple(int(value) for value in config.experiment.eval_inner_steps)
    summaries = [
        runtime.evaluator.evaluate(
            server=runtime.server,
            clients=runtime.clients,
            adaptation_steps=adaptation_steps,
            seed=int(config.seed),
            split=DataSplit.TEST,
        )
        for adaptation_steps in steps
    ]
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "standalone_evaluation.json").write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2),
        encoding="utf-8",
    )
    LOGGER.info("evaluated checkpoint at adaptation steps %s", steps)


if __name__ == "__main__":
    main()
