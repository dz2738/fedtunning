"""Summarize sequential meta-gradient diagnostic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = OmegaConf.load(run_dir / "resolved_config.yaml")
    round_record = json.loads(
        (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    aggregation = round_record["aggregation"]
    return {
        "name": run_dir.name,
        "meta_gradient": str(config.method.inner_loop.meta_gradient),
        "steps": int(config.method.inner_loop.steps),
        "second_order_steps": OmegaConf.select(
            config,
            "method.inner_loop.second_order_steps",
            default=None,
        ),
        "hessian_damping": float(
            OmegaConf.select(
                config,
                "method.inner_loop.hessian_damping",
                default=0.0,
            )
        ),
        "coordinate_lr": float(config.method.inner_loop.coordinate_lr),
        "dtype": str(config.model.dtype),
        "query_loss": aggregation["mean_query_loss"],
        "feedback_norm_mean": aggregation["feedback_norm_mean"],
        "feedback_norm_max": aggregation["feedback_norm_max"],
        "feedback_clipped_clients": aggregation["feedback_clipped_clients"],
        "gradient_norm_before_clip": aggregation["gradient_norm"],
        "gradient_norm_after_clip": aggregation["gradient_norm_after_clip"],
        "second_order_trace": aggregation["second_order_trace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args()
    output_root = arguments.output_root.expanduser().resolve()
    run_dirs = sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and (path / "rounds.jsonl").is_file()
        and (path / "resolved_config.yaml").is_file()
    )
    if not run_dirs:
        raise FileNotFoundError(f"no completed diagnostic runs under {output_root}")
    summaries = [summarize_run(run_dir) for run_dir in run_dirs]
    destination = output_root / "summary.json"
    destination.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
