"""Download, normalize, partition, and summarize configured datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from data.federated_data import build_federated_data

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    data = build_federated_data(config.data, seed=int(config.seed))
    output_dir = Path(str(config.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": int(config.seed),
        "data_config": OmegaConf.to_container(config.data, resolve=True),
        "summary": data.summary(),
        "clients": {
            client_id: {
                "task_id": client.task.task_id,
                "support": len(client.support),
                "query": len(client.query),
                "validation": len(client.validation),
                "test": len(client.test),
            }
            for client_id, client in sorted(data.clients.items())
        },
    }
    destination = output_dir / "data_manifest.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("prepared %d tasks and %d clients", len(data.tasks), len(data.clients))


if __name__ == "__main__":
    main()
