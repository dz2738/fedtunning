"""Dump gold labels and model generations for selected classification tasks."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from data.schema import DataSplit
from trainer.checkpoint import load_checkpoint

try:
    from scripts.train import build_runtime
except ImportError:
    from train import build_runtime

LOGGER = logging.getLogger(__name__)
DEFAULT_TASKS = ("glue_rte", "boolq_qa")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint = config.checkpoint.resume_from
    if not checkpoint:
        raise ValueError("set checkpoint.resume_from")
    requested_tasks = OmegaConf.select(config, "dump_task_ids")
    task_ids = tuple(str(item) for item in (requested_tasks or DEFAULT_TASKS))
    runtime = build_runtime(config)
    load_checkpoint(
        Path(str(checkpoint)).expanduser().resolve(),
        server=runtime.server,
        clients=runtime.clients,
        map_location=runtime.backbone.runtime_device,
    )
    requested_steps = config.experiment.get("dump_adaptation_steps", [0, 5])
    step_list = [int(steps) for steps in requested_steps]
    records = []
    for steps in step_list:
        for client_id, client in sorted(runtime.clients.items()):
            if client.state.task_id not in task_ids:
                continue
            group_id = runtime.server.client_assignments[client_id]
            result = client.evaluate_loss(
                group_id=group_id,
                initial_coordinates=runtime.server.initial_coordinates(client_id),
                subspace=runtime.server.groups[group_id].subspace,
                config=runtime.server.inner_loop,
                adaptation_steps=steps,
                seed=int(config.seed),
                split=DataSplit.TEST,
            )
            golds = list(client.last_golds)
            predictions = list(client.last_predictions)
            record = {
                "client_id": client_id,
                "task_id": client.state.task_id,
                "adaptation_steps": steps,
                "metric": result.metric_value,
                "gold_counts": dict(Counter(golds)),
                "pred_counts": dict(Counter(predictions)),
                "samples": [
                    {"gold": gold, "pred": pred}
                    for gold, pred in list(zip(golds, predictions, strict=True))[:24]
                ],
            }
            records.append(record)
            LOGGER.info(
                "%s steps=%d metric=%.3f gold=%s pred=%s",
                client_id,
                steps,
                result.metric_value if result.metric_value is not None else float("nan"),
                record["gold_counts"],
                record["pred_counts"],
            )

    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "_".join(task_ids) if len(task_ids) <= 3 else "selected_tasks"
    destination = output_dir / f"{stem}_predictions.json"
    payload = {
        "checkpoint": str(checkpoint),
        "adaptation_steps": step_list,
        "task_ids": list(task_ids),
        "clients": records,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", destination)


if __name__ == "__main__":
    main()
