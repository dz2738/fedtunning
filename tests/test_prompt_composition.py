import pytest
import torch

from model.prompt_subspace import (
    PromptSubspace,
    compose_prompt,
    decompose_prompt,
    orthogonality_error,
    orthonormalize_basis,
)


def test_compose_prompt_matches_explicit_sum() -> None:
    torch.manual_seed(10)
    center = torch.randn(3, 5)
    basis = torch.randn(4, 3, 5)
    coordinates = torch.randn(4)
    residual = torch.randn(3, 5)

    actual = compose_prompt(center, basis, coordinates, residual)
    expected = center + sum(
        coordinates[index] * basis[index] for index in range(basis.shape[0])
    ) + residual

    torch.testing.assert_close(actual, expected)


def test_compose_prompt_supports_batched_coordinates() -> None:
    torch.manual_seed(11)
    center = torch.randn(3, 5)
    basis = torch.randn(4, 3, 5)
    coordinates = torch.randn(7, 4)
    residual = torch.randn(7, 3, 5)

    prompt = compose_prompt(center, basis, coordinates, residual)

    assert prompt.shape == (7, 3, 5)
    torch.testing.assert_close(
        prompt[2],
        compose_prompt(center, basis, coordinates[2], residual[2]),
    )


def test_decomposition_recovers_unique_coordinates_and_residual() -> None:
    torch.manual_seed(12)
    center = torch.randn(3, 5, dtype=torch.float64)
    basis = orthonormalize_basis(torch.randn(4, 3, 5, dtype=torch.float64))
    coordinates = torch.randn(6, 4, dtype=torch.float64)
    raw_residual = torch.randn(6, 3, 5, dtype=torch.float64)
    residual = raw_residual - torch.einsum(
        "...k,kld->...ld",
        torch.einsum("dk,...d->...k", basis.reshape(4, -1).T, raw_residual.reshape(6, -1)),
        basis,
    )
    prompt = compose_prompt(center, basis, coordinates, residual)

    decomposition = decompose_prompt(prompt, center, basis, eps=0.0)

    torch.testing.assert_close(decomposition.coordinates, coordinates, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(decomposition.residual, residual, atol=1.0e-10, rtol=1.0e-10)
    torch.testing.assert_close(
        orthogonality_error(decomposition.residual, basis),
        torch.zeros(6, dtype=torch.float64),
        atol=1.0e-10,
        rtol=0.0,
    )


def test_prompt_subspace_owns_only_center_and_basis_state() -> None:
    torch.manual_seed(13)
    subspace = PromptSubspace(prompt_length=3, hidden_size=5, num_basis=4)
    coordinates = torch.randn(4, requires_grad=True)
    residual = subspace.project_residual(torch.randn(3, 5))

    prompt = subspace(coordinates, residual)
    prompt.square().mean().backward()

    assert coordinates.grad is not None
    assert not subspace.center.requires_grad
    assert not subspace.basis.requires_grad
    assert set(subspace.state_dict()) == {"center", "basis"}
    assert float(orthogonality_error(residual, subspace.basis)) < 1.0e-5


def test_set_basis_reorthogonalizes_directions() -> None:
    torch.manual_seed(14)
    subspace = PromptSubspace(prompt_length=3, hidden_size=5, num_basis=4)
    replacement = torch.randn_like(subspace.basis)

    subspace.set_basis(replacement)

    matrix = subspace.basis.reshape(4, -1).T
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(4), atol=1.0e-5, rtol=1.0e-5)


def test_invalid_coordinate_dimension_is_rejected() -> None:
    center = torch.zeros(3, 5)
    basis = torch.randn(4, 3, 5)

    with pytest.raises(ValueError, match="num_basis=4"):
        compose_prompt(center, basis, torch.zeros(3))

