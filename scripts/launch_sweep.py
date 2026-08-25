"""Assign independent experiment processes to a small GPU pool."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunningJob:
    gpu: str
    seed: int
    process: subprocess.Popen[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--script", default="train.py")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def launch(gpu: str, seed: int, args: argparse.Namespace) -> RunningJob:
    script = Path(__file__).with_name(args.script)
    if not script.is_file():
        raise FileNotFoundError(script)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    command = [
        sys.executable,
        str(script),
        f"experiment={args.experiment}",
        f"seed={seed}",
        "device=cuda:0",
        *args.overrides,
    ]
    process = subprocess.Popen(command, env=environment, text=True)
    return RunningJob(gpu=gpu, seed=seed, process=process)


def main() -> None:
    args = parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU identifiers must be unique")
    pending = deque(args.seeds)
    available = deque(args.gpus)
    running: list[RunningJob] = []
    failures: list[tuple[int, int]] = []
    while pending or running:
        while pending and available:
            running.append(launch(available.popleft(), pending.popleft(), args))
        for job in tuple(running):
            return_code = job.process.poll()
            if return_code is None:
                continue
            running.remove(job)
            available.append(job.gpu)
            if return_code != 0:
                failures.append((job.seed, return_code))
        if running:
            running[0].process.wait(timeout=None)
    if failures:
        details = ", ".join(f"seed={seed}:exit={code}" for seed, code in failures)
        raise RuntimeError(f"sweep jobs failed: {details}")


if __name__ == "__main__":
    main()
