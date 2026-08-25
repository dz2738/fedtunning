"""Deterministic single-process federated-round simulation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from client.client import FederatedClient
from server.server import FedTaskPromptServer, ServerRoundSummary
from trainer.evaluator import EvaluationSummary, FederatedEvaluator


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    num_rounds: int = 50
    client_fraction: float = 1.0
    min_clients_per_round: int = 1
    eval_every_rounds: int = 5
    eval_inner_steps: tuple[int, ...] = (5,)

    def __post_init__(self) -> None:
        if self.num_rounds <= 0 or self.min_clients_per_round <= 0:
            raise ValueError("round and minimum-client counts must be positive")
        if not 0.0 < self.client_fraction <= 1.0:
            raise ValueError("client_fraction must be in (0, 1]")
        if self.eval_every_rounds <= 0:
            raise ValueError("eval_every_rounds must be positive")
        if not self.eval_inner_steps or any(step < 0 for step in self.eval_inner_steps):
            raise ValueError("eval_inner_steps must contain non-negative values")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SimulationConfig:
        raw_steps = value.get("eval_inner_steps", (5,))
        return cls(
            num_rounds=int(value.get("num_rounds", 50)),
            client_fraction=float(value.get("client_fraction", 1.0)),
            min_clients_per_round=int(value.get("min_clients_per_round", 1)),
            eval_every_rounds=int(value.get("eval_every_rounds", 5)),
            eval_inner_steps=tuple(int(step) for step in raw_steps),
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    rounds: tuple[ServerRoundSummary, ...]
    evaluations: tuple[EvaluationSummary, ...]


RoundCallback = Callable[[ServerRoundSummary], None]
EvaluationCallback = Callable[[EvaluationSummary], None]


class FederatedSimulator:
    def __init__(
        self,
        *,
        server: FedTaskPromptServer,
        clients: Mapping[str, FederatedClient],
        config: SimulationConfig,
        seed: int,
        evaluator: FederatedEvaluator | None = None,
    ) -> None:
        if not clients:
            raise ValueError("clients cannot be empty")
        self.server = server
        self.clients = dict(clients)
        self.config = config
        self.seed = int(seed)
        self.evaluator = evaluator

    def initialize(self) -> dict[str, str]:
        return self.server.initialize_groups(
            {client_id: client.state for client_id, client in self.clients.items()}
        )

    def sample_clients(self, round_number: int) -> tuple[str, ...]:
        if round_number <= 0:
            raise ValueError("round_number must be positive")
        client_ids = np.asarray(sorted(self.clients), dtype=object)
        sample_size = max(
            self.config.min_clients_per_round,
            math.ceil(len(client_ids) * self.config.client_fraction),
        )
        sample_size = min(sample_size, len(client_ids))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, round_number]))
        selected = rng.choice(client_ids, size=sample_size, replace=False)
        return tuple(sorted(str(client_id) for client_id in selected))

    def run(
        self,
        *,
        on_round: RoundCallback | None = None,
        on_evaluation: EvaluationCallback | None = None,
    ) -> SimulationResult:
        if not self.server.groups:
            self.initialize()
        round_history: list[ServerRoundSummary] = []
        evaluation_history: list[EvaluationSummary] = []
        while self.server.round_number < self.config.num_rounds:
            next_round = self.server.round_number + 1
            summary = self.server.run_round(
                clients=self.clients,
                selected_client_ids=self.sample_clients(next_round),
                seed=self.seed,
            )
            round_history.append(summary)
            if on_round is not None:
                on_round(summary)

            if (
                self.evaluator is not None
                and summary.round_number % self.config.eval_every_rounds == 0
            ):
                for steps in self.config.eval_inner_steps:
                    evaluation = self.evaluator.evaluate(
                        server=self.server,
                        clients=self.clients,
                        adaptation_steps=steps,
                        seed=self.seed,
                    )
                    evaluation_history.append(evaluation)
                    if on_evaluation is not None:
                        on_evaluation(evaluation)

        return SimulationResult(
            rounds=tuple(round_history),
            evaluations=tuple(evaluation_history),
        )
