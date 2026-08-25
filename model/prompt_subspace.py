"""Affine prompt subspace and orthogonal personalization primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn


_LOW_PRECISION_DTYPES: Final = (
    torch.float16,
    torch.bfloat16,
)


def _linear_algebra_dtype(tensor: Tensor) -> torch.dtype:
    """Use FP32 for decompositions unsupported or unstable in low precision."""

    if tensor.dtype in _LOW_PRECISION_DTYPES:
        return torch.float32
    return tensor.dtype


def _compute_dtype(*tensors: Tensor) -> torch.dtype:
    """Choose a common stable dtype for mixed-precision operations."""

    if not tensors:
        raise ValueError("at least one tensor is required")

    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)

    # Prompt-subspace tensors are small. Performing the associated linear
    # algebra in FP32 avoids unstable BF16/FP16 projection and decomposition.
    if dtype in _LOW_PRECISION_DTYPES:
        return torch.float32

    return dtype


def _validate_floating_tensor(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(
            f"{name} must be floating point, got {tensor.dtype}"
        )


def _validate_basis(basis: Tensor) -> tuple[int, int, int]:
    _validate_floating_tensor("basis", basis)

    if basis.ndim != 3:
        raise ValueError(
            "basis must have shape "
            "[num_basis, prompt_length, hidden_size], "
            f"got {basis.shape}"
        )

    num_basis, prompt_length, hidden_size = basis.shape

    if min(num_basis, prompt_length, hidden_size) <= 0:
        raise ValueError("basis dimensions must all be positive")

    if num_basis > prompt_length * hidden_size:
        raise ValueError(
            "num_basis cannot exceed the flattened prompt dimension"
        )

    return num_basis, prompt_length, hidden_size


def basis_matrix(basis: Tensor) -> Tensor:
    """Return Q with shape [prompt_dimension, num_basis]."""

    _validate_basis(basis)

    return basis.reshape(
        basis.shape[0],
        -1,
    ).transpose(0, 1)


def orthonormalize_basis(
    basis: Tensor,
    *,
    rank_tolerance: float | None = None,
) -> Tensor:
    """Orthonormalize flattened directions using reduced QR factorization."""

    num_basis, prompt_length, hidden_size = _validate_basis(basis)

    work_dtype = _linear_algebra_dtype(basis)

    matrix = basis_matrix(basis).to(dtype=work_dtype)

    if rank_tolerance is None:
        rank_tensor = torch.linalg.matrix_rank(matrix)
    else:
        rank_tensor = torch.linalg.matrix_rank(
            matrix,
            tol=rank_tolerance,
        )

    rank = int(rank_tensor.item())

    if rank != num_basis:
        raise ValueError(
            "basis must have full column rank, "
            f"got rank={rank}, K={num_basis}"
        )

    orthonormal, _ = torch.linalg.qr(
        matrix,
        mode="reduced",
    )

    return (
        orthonormal.transpose(0, 1)
        .reshape(
            num_basis,
            prompt_length,
            hidden_size,
        )
        .to(dtype=basis.dtype)
    )


def _validate_prompt_shape(
    name: str,
    tensor: Tensor,
    basis: Tensor,
) -> None:
    """Validate prompt shape and device while allowing mixed precision."""

    _validate_floating_tensor(name, tensor)

    expected = tuple(basis.shape[-2:])

    if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != expected:
        raise ValueError(
            f"{name} must end in shape {expected}, "
            f"got {tuple(tensor.shape)}"
        )

    if tensor.device != basis.device:
        raise ValueError(
            f"{name} and basis must be on the same device"
        )


def _span_coefficients(
    value: Tensor,
    basis: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Solve argmin_c ||Q c - value|| for flattened prompt values."""

    _validate_prompt_shape("value", value, basis)

    if eps < 0:
        raise ValueError("eps must be non-negative")

    num_basis = basis.shape[0]
    prompt_dimension = basis.shape[1] * basis.shape[2]
    leading_shape = value.shape[:-2]

    work_dtype = _compute_dtype(
        value,
        basis,
    )

    matrix = basis_matrix(basis).to(
        dtype=work_dtype,
    )

    flattened = (
        value.reshape(-1, prompt_dimension)
        .transpose(0, 1)
        .to(dtype=work_dtype)
    )

    gram = matrix.transpose(0, 1) @ matrix

    identity = torch.eye(
        num_basis,
        device=gram.device,
        dtype=gram.dtype,
    )

    # Orthonormal bases use the direct projection. The general case solves
    # the regularized normal equation.
    if torch.allclose(
        gram.detach(),
        identity,
        atol=max(1.0e-5, eps * 10),
        rtol=1.0e-5,
    ):
        coefficients = (
            matrix.transpose(0, 1) @ flattened
        )
    else:
        coefficients = torch.linalg.solve(
            gram + eps * identity,
            matrix.transpose(0, 1) @ flattened,
        )

    return coefficients.transpose(0, 1).reshape(
        *leading_shape,
        num_basis,
    )


def project_onto_span(
    value: Tensor,
    basis: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Project prompt-shaped tensors onto span(basis)."""

    coefficients = _span_coefficients(
        value,
        basis,
        eps=eps,
    )

    work_dtype = _compute_dtype(
        value,
        basis,
        coefficients,
    )

    projected = torch.einsum(
        "...k,kld->...ld",
        coefficients.to(dtype=work_dtype),
        basis.to(dtype=work_dtype),
    )

    # Preserve the dtype of the value being projected. In particular,
    # FP32 residual states remain FP32.
    return projected.to(dtype=value.dtype)


def project_onto_orthogonal_complement(
    value: Tensor,
    basis: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    """Apply P_perp without constructing a full projection matrix."""

    return value - project_onto_span(
        value,
        basis,
        eps=eps,
    )


def compose_prompt(
    center: Tensor,
    basis: Tensor,
    coordinates: Tensor,
    residual: Tensor | None = None,
) -> Tensor:
    """Compose B0 + sum_k c_k B_k + U."""

    _validate_basis(basis)
    _validate_prompt_shape(
        "center",
        center,
        basis,
    )

    if center.ndim != 2:
        raise ValueError(
            "center must have shape "
            "[prompt_length, hidden_size], "
            f"got {center.shape}"
        )

    _validate_floating_tensor(
        "coordinates",
        coordinates,
    )

    if (
        coordinates.ndim < 1
        or coordinates.shape[-1] != basis.shape[0]
    ):
        raise ValueError(
            "coordinates must end in "
            f"num_basis={basis.shape[0]}, "
            f"got {coordinates.shape}"
        )

    if coordinates.device != basis.device:
        raise ValueError(
            "coordinates and basis must be on the same device"
        )

    if residual is not None:
        _validate_prompt_shape(
            "residual",
            residual,
            basis,
        )

    compute_tensors = [
        center,
        basis,
        coordinates,
    ]

    if residual is not None:
        compute_tensors.append(residual)

    work_dtype = _compute_dtype(
        *compute_tensors,
    )

    center_work = center.to(
        dtype=work_dtype,
    )

    basis_work = basis.to(
        dtype=work_dtype,
    )

    coordinates_work = coordinates.to(
        dtype=work_dtype,
    )

    shared_update = torch.einsum(
        "...k,kld->...ld",
        coordinates_work,
        basis_work,
    )

    prompt = center_work + shared_update

    if residual is not None:
        try:
            prompt = (
                prompt
                + residual.to(dtype=work_dtype)
            )
        except RuntimeError as error:
            raise ValueError(
                f"residual shape {residual.shape} "
                "cannot broadcast to prompt shape "
                f"{prompt.shape}"
            ) from error

    # The resulting prompt is consumed by the backbone, so it follows the
    # dtype of the basis/center module, typically BF16.
    return prompt.to(dtype=basis.dtype)


@dataclass(frozen=True, slots=True)
class PromptDecomposition:
    coordinates: Tensor
    residual: Tensor


def decompose_prompt(
    prompt: Tensor,
    center: Tensor,
    basis: Tensor,
    *,
    eps: float = 1.0e-6,
) -> PromptDecomposition:
    """Recover coordinates and orthogonal residual for fixed B0 and Q."""

    _validate_prompt_shape(
        "prompt",
        prompt,
        basis,
    )

    _validate_prompt_shape(
        "center",
        center,
        basis,
    )

    if center.ndim != 2:
        raise ValueError(
            "center must be a single prompt matrix"
        )

    work_dtype = _compute_dtype(
        prompt,
        center,
        basis,
    )

    prompt_work = prompt.to(
        dtype=work_dtype,
    )

    center_work = center.to(
        dtype=work_dtype,
    )

    basis_work = basis.to(
        dtype=work_dtype,
    )

    delta = prompt_work - center_work

    coordinates = _span_coefficients(
        delta,
        basis,
        eps=eps,
    ).to(dtype=work_dtype)

    shared_update = torch.einsum(
        "...k,kld->...ld",
        coordinates,
        basis_work,
    )

    residual = delta - shared_update

    return PromptDecomposition(
        coordinates=coordinates,
        residual=residual,
    )


def orthogonality_error(
    residual: Tensor,
    basis: Tensor,
) -> Tensor:
    """Return relative ||Q^T u|| for diagnostics."""

    _validate_prompt_shape(
        "residual",
        residual,
        basis,
    )

    prompt_dimension = (
        basis.shape[1] * basis.shape[2]
    )

    work_dtype = _compute_dtype(
        residual,
        basis,
    )

    matrix = basis_matrix(basis).to(
        dtype=work_dtype,
    )

    flattened = (
        residual.reshape(-1, prompt_dimension)
        .transpose(0, 1)
        .to(dtype=work_dtype)
    )

    overlap = (
        matrix.transpose(0, 1)
        @ flattened
    )

    numerator = torch.linalg.vector_norm(
        overlap,
        dim=0,
    )

    denominator = torch.linalg.vector_norm(
        flattened,
        dim=0,
    ).clamp_min(
        torch.finfo(work_dtype).eps
    )

    return (
        numerator / denominator
    ).reshape(residual.shape[:-2])


class PromptSubspace(nn.Module):
    """Container for one group's center and shared prompt directions.

    The tensors are non-trainable Parameters so they participate in state_dict
    and device/dtype moves. Normal rounds update only the coordinate generator;
    basis maintenance mutates these tensors explicitly under no_grad.
    """

    def __init__(
        self,
        *,
        prompt_length: int,
        hidden_size: int,
        num_basis: int,
        init_std: float = 0.02,
        projection_eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if min(
            prompt_length,
            hidden_size,
            num_basis,
        ) <= 0:
            raise ValueError(
                "prompt_length, hidden_size, "
                "and num_basis must be positive"
            )

        if num_basis > prompt_length * hidden_size:
            raise ValueError(
                "num_basis cannot exceed "
                "flattened prompt dimension"
            )

        if init_std <= 0:
            raise ValueError(
                "init_std must be positive"
            )

        if projection_eps < 0:
            raise ValueError(
                "projection_eps must be non-negative"
            )

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        center = torch.zeros(
            prompt_length,
            hidden_size,
            **factory_kwargs,
        )

        raw_basis = torch.randn(
            num_basis,
            prompt_length,
            hidden_size,
            **factory_kwargs,
        )

        raw_basis.mul_(init_std)

        directions = orthonormalize_basis(
            raw_basis,
        )

        self.center = nn.Parameter(
            center,
            requires_grad=False,
        )

        self.basis = nn.Parameter(
            directions,
            requires_grad=False,
        )

        self.projection_eps = float(
            projection_eps
        )

    @property
    def num_basis(self) -> int:
        return self.basis.shape[0]

    @property
    def prompt_length(self) -> int:
        return self.center.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.center.shape[1]

    @property
    def prompt_dimension(self) -> int:
        return (
            self.prompt_length
            * self.hidden_size
        )

    def forward(
        self,
        coordinates: Tensor,
        residual: Tensor | None = None,
    ) -> Tensor:
        return compose_prompt(
            self.center,
            self.basis,
            coordinates,
            residual,
        )

    def project_residual(
        self,
        residual: Tensor,
    ) -> Tensor:
        return project_onto_orthogonal_complement(
            residual,
            self.basis,
            eps=self.projection_eps,
        )

    def decompose(
        self,
        prompt: Tensor,
    ) -> PromptDecomposition:
        return decompose_prompt(
            prompt,
            self.center,
            self.basis,
            eps=self.projection_eps,
        )

    @torch.no_grad()
    def set_center(
        self,
        center: Tensor,
    ) -> None:
        if center.shape != self.center.shape:
            raise ValueError(
                f"expected center shape {self.center.shape}, "
                f"got {center.shape}"
            )

        self.center.copy_(
            center.to(
                device=self.center.device,
                dtype=self.center.dtype,
            )
        )

    @torch.no_grad()
    def set_basis(
        self,
        basis: Tensor,
        *,
        orthonormalize: bool = True,
    ) -> None:
        if basis.shape != self.basis.shape:
            raise ValueError(
                f"expected basis shape {self.basis.shape}, "
                f"got {basis.shape}"
            )

        value = basis.to(
            device=self.basis.device,
            dtype=self.basis.dtype,
        )

        if orthonormalize:
            value = orthonormalize_basis(value)

        self.basis.copy_(value)