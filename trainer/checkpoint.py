"""Atomic experiment checkpoints including RNG and lightweight client state."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from client.client import FederatedClient
from server.server import FedTaskPromptServer

CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["torch_cuda"]]
        )


def _client_state(clients: Mapping[str, FederatedClient]) -> dict[str, dict[str, Any]]:
    return {
        client_id: {
            "group_id": client.state.group_id,
            "deployment_residual": client.state.deployment_residual,
            "metric_history": list(client.state.metric_history),
        }
        for client_id, client in clients.items()
    }


def save_checkpoint(
    path: str | Path,
    *,
    server: FedTaskPromptServer,
    clients: Mapping[str, FederatedClient],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "server": server.checkpoint_state(),
        "clients": _client_state(clients),
        "rng": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    server: FedTaskPromptServer,
    clients: Mapping[str, FederatedClient],
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load a trusted local checkpoint and reconstruct its dynamic groups."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    saved_clients = payload["clients"]
    if set(saved_clients) != set(clients):
        raise ValueError("checkpoint client IDs differ from the current experiment")
    if not server.groups:
        server.initialize_groups_from_embeddings(
            {client_id: client.state for client_id, client in clients.items()},
            payload["server"]["task_embeddings"],
        )
    server.load_checkpoint_state(payload["server"])
    for client_id, client in clients.items():
        saved = saved_clients[client_id]
        client.state.group_id = saved["group_id"]
        client.state.deployment_residual = saved["deployment_residual"]
        client.state.metric_history = list(saved["metric_history"])
    restore_rng_state(payload["rng"])
    return payload["extra"]
