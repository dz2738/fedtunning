"""Theoretical communication, wall-clock, and peak-memory accounting."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch


class CommunicationDirection(StrEnum):
    DOWNLOAD = "download"
    UPLOAD = "upload"


@dataclass(frozen=True, slots=True)
class CommunicationConfig:
    count_downloads: bool = True
    count_uploads: bool = True
    dtype_bytes: int = 4

    def __post_init__(self) -> None:
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CommunicationConfig:
        return cls(
            count_downloads=bool(value.get("count_downloads", True)),
            count_uploads=bool(value.get("count_uploads", True)),
            dtype_bytes=int(value.get("dtype_bytes", 4)),
        )


@dataclass(frozen=True, slots=True)
class CommunicationEvent:
    round_number: int
    client_id: str
    name: str
    direction: CommunicationDirection
    numel: int
    bytes: int
    maintenance: bool


@dataclass(frozen=True, slots=True)
class CommunicationSummary:
    total_bytes: int
    download_bytes: int
    upload_bytes: int
    normal_round_bytes: int
    maintenance_bytes: int
    average_bytes_per_round: float


class CommunicationLedger:
    def __init__(self, config: CommunicationConfig) -> None:
        self.config = config
        self.events: list[CommunicationEvent] = []

    def record(
        self,
        *,
        round_number: int,
        client_id: str,
        name: str,
        direction: CommunicationDirection | str,
        numel: int,
        maintenance: bool = False,
        dtype_bytes: int | None = None,
    ) -> None:
        direction = CommunicationDirection(direction)
        if round_number <= 0 or numel < 0:
            raise ValueError("round_number must be positive and numel non-negative")
        if direction is CommunicationDirection.DOWNLOAD and not self.config.count_downloads:
            return
        if direction is CommunicationDirection.UPLOAD and not self.config.count_uploads:
            return
        element_bytes = self.config.dtype_bytes if dtype_bytes is None else int(dtype_bytes)
        if element_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        self.events.append(
            CommunicationEvent(
                round_number=round_number,
                client_id=str(client_id),
                name=str(name),
                direction=direction,
                numel=int(numel),
                bytes=int(numel) * element_bytes,
                maintenance=bool(maintenance),
            )
        )

    def record_training_client(
        self,
        *,
        round_number: int,
        client_id: str,
        num_basis: int,
        prompt_dimension: int,
        maintenance: bool,
    ) -> None:
        self.record(
            round_number=round_number,
            client_id=client_id,
            name="initial_coordinates",
            direction=CommunicationDirection.DOWNLOAD,
            numel=num_basis,
        )
        self.record(
            round_number=round_number,
            client_id=client_id,
            name="coordinate_feedback",
            direction=CommunicationDirection.UPLOAD,
            numel=num_basis,
        )
        if maintenance:
            self.record(
                round_number=round_number,
                client_id=client_id,
                name="terminal_residual",
                direction=CommunicationDirection.UPLOAD,
                numel=prompt_dimension,
                maintenance=True,
            )

    def summary(self) -> CommunicationSummary:
        total = sum(event.bytes for event in self.events)
        downloads = sum(
            event.bytes
            for event in self.events
            if event.direction is CommunicationDirection.DOWNLOAD
        )
        uploads = total - downloads
        maintenance = sum(event.bytes for event in self.events if event.maintenance)
        rounds = {event.round_number for event in self.events}
        return CommunicationSummary(
            total_bytes=total,
            download_bytes=downloads,
            upload_bytes=uploads,
            normal_round_bytes=total - maintenance,
            maintenance_bytes=maintenance,
            average_bytes_per_round=total / max(len(rounds), 1),
        )


@dataclass(frozen=True, slots=True)
class RuntimeMeasurement:
    wall_seconds: float
    peak_device_bytes: int


class RuntimeTracker:
    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._start: float | None = None
        self.measurement: RuntimeMeasurement | None = None

    def __enter__(self) -> RuntimeTracker:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._start is None:
            raise RuntimeError("RuntimeTracker was not entered")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
        else:
            peak = 0
        self.measurement = RuntimeMeasurement(
            wall_seconds=time.perf_counter() - self._start,
            peak_device_bytes=peak,
        )
