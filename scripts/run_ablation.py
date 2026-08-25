"""Launch ablation variants that are exactly expressible by current configs."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


LOGGER = logging.getLogger(__name__)


VARIANT_OVERRIDES: dict[str, tuple[str, ...] | None] = {
    "full": (),
    "single_group": ("method.grouping.assignment_threshold=-1.0",),
    "no_local_residual": (
        "method.inner_loop.residual_lr=0.0",
        "method.inner_loop.residual_weight_decay=0.0",
    ),
    "no_orthogonality": ("method.inner_loop.project_residual=false",),
    "first_order": ("method.inner_loop.meta_gradient=first_order",),
    "full_second_order": ("method.inner_loop.meta_gradient=full_second_order",),
    "fixed_basis": ("method.basis_maintenance.enabled=false",),
    "no_center_update": ("method.basis_maintenance.center_update_rate=0.0",),
    "random_group": None,
    "zero_coordinate_init": None,
    "group_mean_init": None,
    "private_only": None,
    "no_direction_replacement": None,
}


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    train_script = Path(__file__).with_name("train.py")
    unsupported = []
    for variant in config.experiment.variants:
        name = str(variant)
        if name not in VARIANT_OVERRIDES:
            raise ValueError(f"unknown ablation variant: {name}")
        overrides = VARIANT_OVERRIDES[name]
        if overrides is None:
            unsupported.append(name)
            LOGGER.warning("skip unsupported exact ablation: %s", name)
            continue
        command = [
            sys.executable,
            str(train_script),
            "experiment=ablation",
            f"experiment.name=ablation/{name}",
            f"seed={int(config.seed)}",
            *overrides,
        ]
        subprocess.run(command, check=True)
    if unsupported:
        LOGGER.warning(
            "unsupported variants require additional algorithm switches: %s",
            ", ".join(unsupported),
        )


if __name__ == "__main__":
    main()
