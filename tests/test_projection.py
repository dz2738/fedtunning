import pytest
import torch

from client.client import _clip_for_upload
from client.state import ClientRoundResult, MetaGradientMode
from model.prompt_subspace import (
    basis_matrix,
    orthogonality_error,
    orthonormalize_basis,
    project_onto_orthogonal_complement,
    project_onto_span,
)
from server.basis_maintenance import BasisMaintainer, BasisMaintenanceConfig
from server.group_state import GroupState


def test_orthonormalize_basis_produces_identity_gram() -> None:
    torch.manual_seed(1)
    basis = torch.randn(4, 3, 5, dtype=torch.float64)

    orthonormal = orthonormalize_basis(basis)
    matrix = basis_matrix(orthonormal)

    torch.testing.assert_close(
        matrix.T @ matrix,
        torch.eye(4, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_general_full_rank_projection_is_orthogonal_and_reconstructs() -> None:
    torch.manual_seed(2)
    basis = torch.randn(3, 4, 6, dtype=torch.float64)
    value = torch.randn(5, 4, 6, dtype=torch.float64)

    parallel = project_onto_span(value, basis, eps=0.0)
    orthogonal = project_onto_orthogonal_complement(value, basis, eps=0.0)

    torch.testing.assert_close(parallel + orthogonal, value)
    torch.testing.assert_close(
        basis_matrix(basis).T @ orthogonal.reshape(5, -1).T,
        torch.zeros(3, 5, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        orthogonality_error(orthogonal, basis),
        torch.zeros(5, dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    )


def test_projection_is_idempotent_for_orthonormal_basis() -> None:
    torch.manual_seed(3)
    basis = orthonormalize_basis(torch.randn(4, 3, 7))
    value = torch.randn(2, 3, 7)

    once = project_onto_orthogonal_complement(value, basis)
    twice = project_onto_orthogonal_complement(once, basis)

    torch.testing.assert_close(twice, once, atol=2.0e-6, rtol=2.0e-6)


def test_projection_keeps_autograd_path() -> None:
    torch.manual_seed(4)
    basis = orthonormalize_basis(torch.randn(2, 3, 4))
    value = torch.randn(3, 4, requires_grad=True)

    projected = project_onto_orthogonal_complement(value, basis)
    projected.square().sum().backward()

    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_low_precision_qr_returns_original_dtype() -> None:
    torch.manual_seed(5)
    basis = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    orthonormal = orthonormalize_basis(basis)

    assert orthonormal.dtype is torch.bfloat16
    gram = basis_matrix(orthonormal.float()).T @ basis_matrix(orthonormal.float())
    torch.testing.assert_close(gram, torch.eye(2), atol=5.0e-3, rtol=5.0e-3)


def test_rank_deficient_basis_is_rejected() -> None:
    direction = torch.randn(1, 3, 4)
    basis = torch.cat((direction, direction), dim=0)

    with pytest.raises(ValueError, match="full column rank"):
        orthonormalize_basis(basis)


def test_basis_maintenance_replaces_the_least_used_direction() -> None:
    torch.manual_seed(31)
    group = GroupState(
        group_id="group_0000",
        semantic_prototype=torch.tensor([1.0, 0.0]),
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    candidate = group.subspace.project_residual(torch.randn_like(group.subspace.center))
    candidate = candidate / candidate.norm()

    def result(client_id: str, scale: float) -> ClientRoundResult:
        coordinates = torch.tensor([2.0, 0.01], dtype=torch.float64)
        residual = scale * candidate
        return ClientRoundResult(
            client_id=client_id,
            group_id=group.group_id,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
            coordinate_feedback=torch.zeros(2, dtype=torch.float64),
            initial_coordinates=coordinates,
            terminal_coordinates=coordinates,
            terminal_residual=residual,
            residual_energy=float(residual.square().sum()),
            mean_support_loss=1.0,
            query_loss=1.0,
            support_examples=1,
            query_examples=1,
        )

    results = (result("client_a", 1.0), result("client_b", 0.8))
    maintainer = BasisMaintainer(
        BasisMaintenanceConfig(
            interval=1,
            residual_energy_threshold=0.0,
            center_update_rate=0.0,
            explained_energy_ratio=0.9,
            max_replaced_basis=1,
            usage_window=1,
        )
    )
    maintainer.record_coordinate_usage(results)
    summary = maintainer.maintain_group(group, results)

    assert summary.triggered
    assert summary.replaced_indices == (1,)
    flattened = group.subspace.basis.reshape(2, -1)
    gram = flattened @ flattened.T
    torch.testing.assert_close(
        gram,
        torch.eye(2, dtype=gram.dtype),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_client_upload_clipping_does_not_mutate_raw_residual() -> None:
    residual = torch.tensor([[3.0, 4.0]])

    uploaded = _clip_for_upload(residual, 2.0)

    torch.testing.assert_close(residual, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(uploaded.norm(), torch.tensor(2.0))
    assert not uploaded.requires_grad


def test_relative_residual_energy_controls_maintenance_trigger() -> None:
    group = GroupState(
        group_id="group_0000",
        semantic_prototype=torch.tensor([1.0, 0.0]),
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    residual = group.subspace.project_residual(torch.randn(2, 3, dtype=torch.float64))
    residual = residual / residual.norm()
    coordinates = torch.zeros(2, dtype=torch.float64)
    result = ClientRoundResult(
        client_id="client_a",
        group_id=group.group_id,
        meta_gradient=MetaGradientMode.FIRST_ORDER,
        coordinate_feedback=torch.zeros(2, dtype=torch.float64),
        initial_coordinates=coordinates,
        terminal_coordinates=coordinates,
        terminal_residual=residual,
        residual_energy=1.0,
        mean_support_loss=1.0,
        query_loss=1.0,
        support_examples=1,
        query_examples=1,
        prompt_energy=100.0,
    )

    summary = BasisMaintainer(
        BasisMaintenanceConfig(
            interval=1,
            energy_mode="relative",
            residual_energy_threshold=0.1,
        )
    ).maintain_group(group, (result,))

    assert not summary.triggered
    assert summary.residual_energy == 1.0
    assert summary.relative_residual_energy == 0.01


def test_maintenance_updates_center_then_svd_uses_centered_residuals() -> None:
    torch.manual_seed(32)
    group = GroupState(
        group_id="group_0000",
        semantic_prototype=torch.tensor([1.0, 0.0]),
        prompt_length=2,
        hidden_size=4,
        num_basis=2,
        dtype=torch.float64,
    )
    raw = group.subspace.project_residual(torch.randn(2, 2, 4, dtype=torch.float64))
    complement, _ = torch.linalg.qr(raw.reshape(2, -1).T, mode="reduced")
    mean = complement[:, 0].reshape(2, 4)
    variation = complement[:, 1].reshape(2, 4)

    def result(client_id: str, residual: torch.Tensor) -> ClientRoundResult:
        coordinates = torch.tensor([3.0, 0.01], dtype=torch.float64)
        return ClientRoundResult(
            client_id=client_id,
            group_id=group.group_id,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
            coordinate_feedback=torch.zeros(2, dtype=torch.float64),
            initial_coordinates=coordinates,
            terminal_coordinates=coordinates,
            terminal_residual=residual,
            residual_energy=float(residual.square().sum()),
            mean_support_loss=1.0,
            query_loss=1.0,
            support_examples=1,
            query_examples=1,
        )

    results = (
        result("client_a", mean + 0.5 * variation),
        result("client_b", mean - 0.5 * variation),
    )
    maintainer = BasisMaintainer(
        BasisMaintenanceConfig(
            interval=1,
            residual_energy_threshold=0.0,
            center_update_rate=0.25,
            explained_energy_ratio=0.9,
            max_replaced_basis=1,
            usage_window=1,
        )
    )
    maintainer.record_coordinate_usage(results)
    summary = maintainer.maintain_group(group, results)

    torch.testing.assert_close(group.subspace.center, 0.25 * mean)
    learned = group.subspace.basis[1].reshape(-1)
    assert abs(float(learned @ variation.reshape(-1))) > 1.0 - 1.0e-6
    assert summary.replaced_indices == (1,)


def test_identical_residuals_update_center_without_replacing_basis() -> None:
    torch.manual_seed(33)
    group = GroupState(
        group_id="group_0000",
        semantic_prototype=torch.tensor([1.0, 0.0]),
        prompt_length=2,
        hidden_size=3,
        num_basis=2,
        dtype=torch.float64,
    )
    residual = group.subspace.project_residual(torch.randn(2, 3, dtype=torch.float64))
    old_basis = group.subspace.basis.detach().clone()

    def result(client_id: str) -> ClientRoundResult:
        coordinates = torch.zeros(2, dtype=torch.float64)
        return ClientRoundResult(
            client_id=client_id,
            group_id=group.group_id,
            meta_gradient=MetaGradientMode.COORDINATE_SECOND_ORDER,
            coordinate_feedback=torch.zeros(2, dtype=torch.float64),
            initial_coordinates=coordinates,
            terminal_coordinates=coordinates,
            terminal_residual=residual,
            residual_energy=float(residual.square().sum()),
            mean_support_loss=1.0,
            query_loss=1.0,
            support_examples=1,
            query_examples=1,
        )

    summary = BasisMaintainer(
        BasisMaintenanceConfig(
            interval=1,
            residual_energy_threshold=0.0,
            center_update_rate=0.5,
            max_replaced_basis=1,
        )
    ).maintain_group(group, (result("a"), result("b")))

    assert summary.triggered
    assert summary.replaced_indices == ()
    torch.testing.assert_close(group.subspace.center, 0.5 * residual)
    torch.testing.assert_close(group.subspace.basis, old_basis)
