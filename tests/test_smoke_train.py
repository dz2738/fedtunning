"""End-to-end smoke test without downloading a language model."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from client.inner_loop import adapt_and_compute_feedback
from client.state import (
    ClientEvaluationResult,
    ClientRoundResult,
    ClientState,
    InnerLoopConfig,
    MetaGradientMode,
)
from data.schema import TaskDescription
from model.coordinate_generator import CoordinateGenerator, CoordinateGeneratorConfig
from model.prompt_subspace import PromptSubspace
from server.aggregation import ServerOptimizerConfig
from server.basis_maintenance import BasisMaintenanceConfig
from server.grouping import GroupingConfig
from server.server import FedTaskPromptServer
from trainer.evaluator import FederatedEvaluator
from trainer.simulator import FederatedSimulator, SimulationConfig


class FakeTaskEncoder(nn.Module):
    """Satisfy the server contract when embeddings are supplied directly."""

    embedding_dim = 2

    def clear_cache(self) -> None:
        pass


class SyntheticClient:
    """Exercise local adaptation while sharing one inert backbone sentinel."""

    def __init__(
        self,
        client_id: str,
        task_id: str,
        target: Tensor,
        shared_backbone: object,
    ) -> None:
        self.state = ClientState(
            client_id=client_id,
            task_id=task_id,
            description=TaskDescription(
                op=f"operation_{task_id}",
                input_object="synthetic_input",
                output_format="synthetic_output",
                domain="synthetic",
                language="en",
            ),
        )
        self.target = target
        self.backbone = shared_backbone

    def _losses(self):
        def support_loss(prompt: Tensor, step: int) -> Tensor:
            scale = 1.0 + 0.05 * step
            return 0.5 * scale * (prompt - self.target).square().mean()

        def query_loss(prompt: Tensor) -> Tensor:
            return 0.5 * (prompt - 0.9 * self.target).square().mean()

        return support_loss, query_loss

    def run_round(
        self,
        *,
        group_id: str,
        initial_coordinates: Tensor,
        subspace: PromptSubspace,
        config: InnerLoopConfig,
        round_seed: int,
        return_residual: bool,
    ) -> ClientRoundResult:
        del round_seed
        self.state.assign_group(group_id)
        support_loss, query_loss = self._losses()
        adaptation = adapt_and_compute_feedback(
            support_loss=support_loss,
            query_loss=query_loss,
            subspace=subspace,
            initial_coordinates=initial_coordinates,
            config=config,
        )
        residual_energy = float(adaptation.terminal_residual.square().sum())
        return ClientRoundResult(
            client_id=self.state.client_id,
            group_id=group_id,
            meta_gradient=config.meta_gradient,
            coordinate_feedback=adaptation.coordinate_feedback,
            initial_coordinates=adaptation.initial_coordinates,
            terminal_coordinates=adaptation.terminal_coordinates,
            terminal_residual=adaptation.terminal_residual if return_residual else None,
            residual_energy=residual_energy,
            mean_support_loss=adaptation.mean_support_loss,
            query_loss=adaptation.query_loss,
            support_examples=4,
            query_examples=3,
        )

    def evaluate_loss(
        self,
        *,
        group_id: str,
        initial_coordinates: Tensor,
        subspace: PromptSubspace,
        config: InnerLoopConfig,
        adaptation_steps: int,
        seed: int,
    ) -> ClientEvaluationResult:
        del seed
        support_loss, query_loss = self._losses()
        if adaptation_steps == 0:
            residual = torch.zeros_like(subspace.center)
            loss = query_loss(subspace(initial_coordinates.detach(), residual))
            mean_support_loss = None
            residual_energy = 0.0
        else:
            evaluation_config = replace(
                config,
                steps=adaptation_steps,
                meta_gradient=MetaGradientMode.FIRST_ORDER,
            )
            adaptation = adapt_and_compute_feedback(
                support_loss=support_loss,
                query_loss=query_loss,
                subspace=subspace,
                initial_coordinates=initial_coordinates,
                config=evaluation_config,
            )
            loss = torch.tensor(adaptation.query_loss)
            mean_support_loss = adaptation.mean_support_loss
            residual_energy = float(adaptation.terminal_residual.square().sum())
        return ClientEvaluationResult(
            client_id=self.state.client_id,
            group_id=group_id,
            adaptation_steps=adaptation_steps,
            mean_support_loss=mean_support_loss,
            test_loss=float(loss.detach()),
            residual_energy=residual_energy,
            support_examples=4,
            test_examples=2,
        )


def test_two_group_training_uses_one_generator_and_one_backbone() -> None:
    torch.manual_seed(101)
    shared_backbone = object()
    targets = {
        "client_a0": torch.full((2, 3), 0.5),
        "client_a1": torch.full((2, 3), 0.7),
        "client_b0": torch.full((2, 3), -0.4),
        "client_b1": torch.full((2, 3), -0.6),
    }
    clients = {
        client_id: SyntheticClient(
            client_id,
            task_id="task_a" if "_a" in client_id else "task_b",
            target=target,
            shared_backbone=shared_backbone,
        )
        for client_id, target in targets.items()
    }
    generator = CoordinateGenerator(
        CoordinateGeneratorConfig(
            embedding_dim=2,
            num_basis=2,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        )
    )
    server = FedTaskPromptServer(
        task_encoder=FakeTaskEncoder(),
        coordinate_generator=generator,
        grouping=GroupingConfig(assignment_threshold=0.9),
        optimizer=ServerOptimizerConfig(
            lr=0.02,
            weight_decay=0.0,
            max_grad_norm=100.0,
            coordinate_regularization=0.0,
            group_weighting="uniform",
            client_weighting="query_examples",
        ),
        inner_loop=InnerLoopConfig(
            steps=2,
            coordinate_lr=0.1,
            residual_lr=0.05,
            coordinate_weight_decay=0.0,
            residual_weight_decay=0.0,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
        ),
        basis_maintenance=BasisMaintenanceConfig(enabled=False),
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        prompt_dtype=torch.float32,
    )
    embeddings = {
        "client_a0": torch.tensor([1.0, 0.0]),
        "client_a1": torch.tensor([0.99, 0.05]),
        "client_b0": torch.tensor([0.0, 1.0]),
        "client_b1": torch.tensor([0.05, 0.99]),
    }
    server.initialize_groups_from_embeddings(
        {client_id: client.state for client_id, client in clients.items()},
        embeddings,
    )
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in generator.named_parameters()
    }
    simulator = FederatedSimulator(
        server=server,
        clients=clients,
        config=SimulationConfig(
            num_rounds=2,
            client_fraction=1.0,
            min_clients_per_round=1,
            eval_every_rounds=2,
            eval_inner_steps=(0, 1),
        ),
        seed=101,
        evaluator=FederatedEvaluator(worst_client_fraction=0.25),
    )

    result = simulator.run()

    assert len(server.groups) == 2
    assert len(result.rounds) == 2
    assert len(result.evaluations) == 2
    assert all(len(summary.client_results) == 4 for summary in result.evaluations)
    assert all(
        torch.isfinite(torch.tensor(summary.mean_test_loss))
        for summary in result.evaluations
    )
    assert len({id(client.backbone) for client in clients.values()}) == 1
    assert sum(
        isinstance(module, CoordinateGenerator) for module in server.modules()
    ) == 1
    assert any(
        not torch.allclose(parameter, initial_parameters[name])
        for name, parameter in generator.named_parameters()
    )