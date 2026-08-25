import pytest
import torch

from model.prompt_subspace import (
    basis_matrix,
    orthogonality_error,
    orthonormalize_basis,
    project_onto_orthogonal_complement,
    project_onto_span,
)


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

