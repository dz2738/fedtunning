from dataclasses import replace
from itertools import pairwise

import torch

from client.inner_loop import adapt_and_compute_feedback
from client.state import InnerLoopConfig, MetaGradientMode
from model.prompt_subspace import PromptSubspace


def _quadratic_losses(subspace: PromptSubspace):
    target_support = subspace.center + 0.4 * subspace.basis[0] - 0.2 * subspace.basis[1]
    target_query = subspace.center - 0.1 * subspace.basis[0] + 0.3 * subspace.basis[1]

    def support_loss(prompt: torch.Tensor, step: int) -> torch.Tensor:
        del step
        return 0.5 * (prompt - target_support).square().sum()

    def query_loss(prompt: torch.Tensor) -> torch.Tensor:
        return 0.5 * (prompt - target_query).square().sum()

    return support_loss, query_loss


def test_coordinate_second_order_matches_full_when_shared_and_private_decouple() -> None:
    torch.manual_seed(21)
    subspace = PromptSubspace(
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    support_loss, query_loss = _quadratic_losses(subspace)
    initial = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    base = InnerLoopConfig(
        steps=3,
        coordinate_lr=0.1,
        residual_lr=0.1,
        coordinate_weight_decay=0.0,
        residual_weight_decay=0.0,
    )

    coordinate_result = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(base, meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER),
    )
    full_result = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(base, meta_gradient=MetaGradientMode.FULL_SECOND_ORDER),
    )

    torch.testing.assert_close(
        coordinate_result.coordinate_feedback,
        full_result.coordinate_feedback,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_first_order_omits_inner_hessian_factor() -> None:
    torch.manual_seed(22)
    subspace = PromptSubspace(
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    support_loss, query_loss = _quadratic_losses(subspace)
    initial = torch.tensor([0.2, -0.3], dtype=torch.float64)
    base = InnerLoopConfig(
        steps=2,
        coordinate_lr=0.2,
        residual_lr=0.1,
        coordinate_weight_decay=0.0,
        residual_weight_decay=0.0,
    )

    first_order = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(base, meta_gradient=MetaGradientMode.FIRST_ORDER),
    )
    second_order = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(base, meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER),
    )

    assert not torch.allclose(
        first_order.coordinate_feedback,
        second_order.coordinate_feedback,
    )


def test_fixed_support_monitor_records_initial_and_post_update_losses() -> None:
    torch.manual_seed(26)
    subspace = PromptSubspace(
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    support_loss, query_loss = _quadratic_losses(subspace)
    result = adapt_and_compute_feedback(
        support_loss=support_loss,
        support_monitor_loss=lambda prompt: support_loss(prompt, 0),
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=torch.tensor([0.2, -0.3], dtype=torch.float64),
        config=InnerLoopConfig(
            steps=3,
            coordinate_lr=0.1,
            residual_lr=0.1,
            coordinate_weight_decay=0.0,
            residual_weight_decay=0.0,
        ),
    )

    assert len(result.support_monitor_losses) == 4
    assert all(
        later < earlier
        for earlier, later in pairwise(result.support_monitor_losses)
    )


def test_truncated_coordinate_second_order_uses_only_requested_reverse_steps() -> None:
    torch.manual_seed(24)
    subspace = PromptSubspace(
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    support_loss, query_loss = _quadratic_losses(subspace)
    initial = torch.tensor([0.2, -0.3], dtype=torch.float64)
    base = InnerLoopConfig(
        steps=3,
        coordinate_lr=0.1,
        residual_lr=0.1,
        coordinate_weight_decay=0.0,
        residual_weight_decay=0.0,
    )
    first_order = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(base, meta_gradient=MetaGradientMode.FIRST_ORDER),
    )
    truncated = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(
            base,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
            second_order_steps=1,
        ),
    )

    torch.testing.assert_close(
        truncated.coordinate_feedback,
        0.9 * first_order.coordinate_feedback,
    )


def test_hessian_damping_applies_tikhonov_shift_to_coordinate_recurrence() -> None:
    torch.manual_seed(25)
    subspace = PromptSubspace(
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    support_loss, query_loss = _quadratic_losses(subspace)
    initial = torch.tensor([0.2, -0.3], dtype=torch.float64)
    base = InnerLoopConfig(
        steps=1,
        coordinate_lr=0.2,
        residual_lr=0.1,
        coordinate_weight_decay=0.0,
        residual_weight_decay=0.0,
    )
    first_order = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(base, meta_gradient=MetaGradientMode.FIRST_ORDER),
    )
    damped = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=query_loss,
        subspace=subspace,
        initial_coordinates=initial,
        config=replace(
            base,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
            hessian_damping=2.0,
        ),
    )

    torch.testing.assert_close(
        damped.coordinate_feedback,
        0.4 * first_order.coordinate_feedback,
    )


def test_local_residual_remains_orthogonal_after_every_update() -> None:
    torch.manual_seed(23)
    subspace = PromptSubspace(prompt_length=3, hidden_size=4, num_basis=2)
    target = torch.randn_like(subspace.center)

    def support_loss(prompt: torch.Tensor, step: int) -> torch.Tensor:
        del step
        return (prompt - target).square().mean()

    result = adapt_and_compute_feedback(
        support_loss=support_loss,
        query_loss=lambda prompt: (prompt - target).square().mean(),
        subspace=subspace,
        initial_coordinates=torch.zeros(2),
        config=InnerLoopConfig(
            steps=3,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
        ),
    )

    overlap = subspace.basis.reshape(2, -1) @ result.terminal_residual.reshape(-1)
    torch.testing.assert_close(overlap, torch.zeros_like(overlap), atol=1.0e-5, rtol=0.0)


def test_coordinate_feedback_is_detached_for_server_vjp() -> None:
    subspace = PromptSubspace(prompt_length=2, hidden_size=3, num_basis=2)

    result = adapt_and_compute_feedback(
        support_loss=lambda prompt, step: prompt.square().mean(),
        query_loss=lambda prompt: prompt.square().mean(),
        subspace=subspace,
        initial_coordinates=torch.ones(2, requires_grad=True),
        config=InnerLoopConfig(steps=1),
    )

    assert not result.coordinate_feedback.requires_grad
    assert result.coordinate_feedback.shape == (2,)

