"""E2: scale the number of tasks or the number of clients per task.

The two axes are separate. ``--sweep clients`` keeps the 6-task ``scale_t6`` mix
and varies ``data.clients_per_dataset``. ``--sweep tasks`` uses ``scale_t3``,
``scale_t6``, and the 7-task ``prototype`` at the default 2 clients per task.
"""

from __future__ import annotations

import argparse
from datetime import datetime

try:
    from experiment_jobs import ExperimentJob, python_script, run_job_pool
except ImportError:
    from scripts.experiment_jobs import ExperimentJob, python_script, run_job_pool

CLIENT_COUNTS = (2, 4, 8)
TASK_COUNTS = (3, 6, 7)
# data, public_data, target_num_groups, public checkpoint, num_basis
# T=3 type-level M=3 cannot use K=4; T=6 uses K=4; T=7 uses K=6 so yes/no and QA share rank.
TASK_CONFIGS = {
    3: ("scale_t3", "scale_t3_proxy", 2, "artifacts/public_initialization_t3.pt", 3),
    6: ("scale_t6", "scale_t6_proxy", 4, "artifacts/public_initialization_t6.pt", 4),
    7: ("prototype", "prototype_proxy", 5, "artifacts/public_initialization.pt", 6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--sweep",
        nargs="+",
        choices=("clients", "tasks"),
        default=["clients", "tasks"],
    )
    parser.add_argument("--client-counts", nargs="+", type=int, default=list(CLIENT_COUNTS))
    parser.add_argument("--task-counts", nargs="+", type=int, default=list(TASK_COUNTS))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def _client_job(count: int, seed: int, args: argparse.Namespace) -> ExperimentJob:
    run_id = f"clients{count}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"outputs/e2/clients_{count}/seed_{seed}/{run_id}"
    command = python_script(
        "train.py",
        "experiment=e2",
        f"seed={seed}",
        "device=cuda:0",
        f"run_id={run_id}",
        f"output_dir={output_dir}",
        f"data={TASK_CONFIGS[6][0]}",
        f"public_data={TASK_CONFIGS[6][1]}",
        f"data.clients_per_dataset={count}",
        f"method.grouping.target_num_groups={TASK_CONFIGS[6][2]}",
        f"method.public_initialization.checkpoint_path={TASK_CONFIGS[6][3]}",
        f"method.num_basis={TASK_CONFIGS[6][4]}",
        *args.overrides,
    )
    return ExperimentJob(label=f"clients={count}:seed={seed}", command=command)


def _task_job(count: int, seed: int, args: argparse.Namespace) -> ExperimentJob:
    if count not in TASK_CONFIGS:
        raise ValueError(f"unsupported task count {count}; expected {tuple(TASK_CONFIGS)}")
    data_name, public_name, target_groups, checkpoint, num_basis = TASK_CONFIGS[count]
    run_id = f"tasks{count}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"outputs/e2/tasks_{count}/seed_{seed}/{run_id}"
    command = python_script(
        "train.py",
        "experiment=e2",
        f"seed={seed}",
        "device=cuda:0",
        f"run_id={run_id}",
        f"output_dir={output_dir}",
        f"data={data_name}",
        f"public_data={public_name}",
        f"method.grouping.target_num_groups={target_groups}",
        f"method.public_initialization.checkpoint_path={checkpoint}",
        f"method.num_basis={num_basis}",
        *args.overrides,
    )
    return ExperimentJob(label=f"tasks={count}:seed={seed}", command=command)


def main() -> None:
    args = parse_args()
    jobs: list[ExperimentJob] = []
    for seed in args.seeds:
        if "clients" in args.sweep:
            jobs.extend(_client_job(count, seed, args) for count in args.client_counts)
        if "tasks" in args.sweep:
            jobs.extend(_task_job(count, seed, args) for count in args.task_counts)
    run_job_pool(jobs, gpus=args.gpus)


if __name__ == "__main__":
    main()
