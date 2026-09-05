"""Assign independent experiment processes to a small GPU pool."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentJob:
    label: str
    command: tuple[str, ...]
    env: Mapping[str, str] | None = None


@dataclass(slots=True)
class _RunningJob:
    gpu: str
    label: str
    process: subprocess.Popen[str]


def run_job_pool(
    jobs: Sequence[ExperimentJob],
    *,
    gpus: Sequence[str],
) -> None:
    if not jobs:
        raise ValueError("job list cannot be empty")
    if not gpus:
        raise ValueError("at least one GPU identifier is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU identifiers must be unique")
    pending = deque(jobs)
    available = deque(str(gpu) for gpu in gpus)
    running: list[_RunningJob] = []
    failures: list[tuple[str, int]] = []
    while pending or running:
        while pending and available:
            job = pending.popleft()
            gpu = available.popleft()
            environment = os.environ.copy()
            if job.env:
                environment.update(job.env)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(job.command, env=environment, text=True)
            running.append(_RunningJob(gpu=gpu, label=job.label, process=process))
        for active in tuple(running):
            return_code = active.process.poll()
            if return_code is None:
                continue
            running.remove(active)
            available.append(active.gpu)
            if return_code != 0:
                failures.append((active.label, return_code))
        if running:
            running[0].process.wait(timeout=None)
        elif pending:
            time.sleep(0.1)
    if failures:
        details = ", ".join(f"{label}:exit={code}" for label, code in failures)
        raise RuntimeError(f"experiment jobs failed: {details}")


def python_script(script_name: str, *hydra_overrides: str) -> tuple[str, ...]:
    script = Path(__file__).with_name(script_name).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    return (sys.executable, str(script), *hydra_overrides)
