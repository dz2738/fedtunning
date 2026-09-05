"""E1: lock grouping, then compare FedTaskPrompt against prompt baselines.

Default methods are Local Prompt and FedAvg-Prompt. Pass ``--methods`` to add
IFCA, Shared-Local, Per-FedAvg, or FedTaskPrompt. Hydra overrides go after ``--``.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from trainer.baseline_loop import BASELINE_METHODS

try:
    from experiment_jobs import ExperimentJob, python_script, run_job_pool
except ImportError:
    from scripts.experiment_jobs import ExperimentJob, python_script, run_job_pool

DEFAULT_METHODS = ("local_prompt", "fedavg_prompt")
ALL_METHODS = (*BASELINE_METHODS, "fedtaskprompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--experiment", default="e1")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _job(method: str, seed: int, args: argparse.Namespace) -> ExperimentJob:
    run_id = f"{method}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"outputs/e1/{method}/seed_{seed}/{run_id}"
    shared = [
        f"experiment={args.experiment}",
        f"seed={seed}",
        "device=cuda:0",
        f"run_id={run_id}",
        f"output_dir={output_dir}",
        *args.overrides,
    ]
    if method == "fedtaskprompt":
        command = python_script("train.py", *shared)
    else:
        command = python_script(
            "run_baseline.py",
            *shared,
            f"experiment.baseline_method={method}",
        )
    return ExperimentJob(label=f"{method}:seed={seed}", command=command)


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.methods if name not in ALL_METHODS]
    if unknown:
        raise ValueError(f"unsupported E1 methods: {unknown}")
    jobs = [
        _job(method, seed, args)
        for seed in args.seeds
        for method in args.methods
    ]
    run_job_pool(jobs, gpus=args.gpus)


if __name__ == "__main__":
    main()
