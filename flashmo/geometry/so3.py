"""Differentiable SO(3) operations with column-vector rotation conventions.

Rotation vectors have shape ``(..., 3)``; matrices have shape ``(..., 3, 3)``.
Angular velocities are body coordinates: dR/dt = R @ hat(omega).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def hat(vector: Tensor) -> Tensor:
    """Map a rotation vector to its skew-symmetric matrix."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *vector.shape[:-1], 3, 3
    )


def vee(matrix: Tensor) -> Tensor:
    """Inverse of :func:`hat` on skew matrices, with skew projection otherwise."""
    return torch.stack(
        (matrix[..., 2, 1] - matrix[..., 1, 2],
         matrix[..., 0, 2] - matrix[..., 2, 0],
         matrix[..., 1, 0] - matrix[..., 0, 1]), -1
    ) * 0.5


def _rodrigues_coefficients(vector: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    theta2 = vector.square().sum(-1, keepdim=True)
    # Safe values in the inactive branch are essential for finite gradients at 0.
    theta = theta2.clamp_min(1e-12).sqrt()
    threshold = 1e-2 if vector.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-6
    a = torch.where(theta2 < threshold,
                    1 - theta2 / 6 + theta2.square() / 120,
                    torch.sin(theta) / theta)
    b = torch.where(theta2 < threshold,
                    0.5 - theta2 / 24 + theta2.square() / 720,
                    (1 - torch.cos(theta)) / theta.square())
    c = torch.where(theta2 < threshold,
                    1 / 6 - theta2 / 120 + theta2.square() / 5040,
                    (theta - torch.sin(theta)) / theta.pow(3))
    return a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)


def so3_exp(vector: Tensor) -> Tensor:
    """Rodrigues exponential, including a Taylor branch at the identity."""
    a, b, _ = _rodrigues_coefficients(vector)
    skew = hat(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + a * skew + b * (skew @ skew)


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Return a unit quaternion (real component first) with nonnegative real part.

    Selecting the largest quaternion component avoids division by sin(theta),
    including near pi. At the pi cut locus the axis sign is inherently ambiguous.
    """
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    squared = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                           1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    magnitudes = squared.clamp_min(torch.finfo(matrix.dtype).eps).sqrt()
    candidates = torch.stack((
        torch.stack((squared[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, squared[..., 1], m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, squared[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, squared[..., 3]), -1),
    ), -2) / (2 * magnitudes[..., :, None])
    index = magnitudes.argmax(-1)
    quaternion = candidates.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    quaternion = F.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def so3_log(matrix: Tensor) -> Tensor:
    """Principal logarithm with angles in [0, pi] and finite identity gradients."""
    quaternion = matrix_to_quaternion(matrix)
    imaginary = quaternion[..., 1:]
    sin_half_squared = imaginary.square().sum(-1, keepdim=True)
    sin_half = sin_half_squared.clamp_min(1e-16).sqrt()
    angle = 2 * torch.atan2(sin_half, quaternion[..., :1])
    scale = torch.where(sin_half_squared < 1e-8,
                        2 + sin_half_squared / 3 + 3 * sin_half_squared.square() / 20,
                        angle / sin_half)
    return imaginary * scale


def so3_right_jacobian(vector: Tensor) -> Tensor:
    """J_r such that vee(Exp(a)^T d Exp(a)/dt) = J_r(a) da/dt."""
    _, b, c = _rodrigues_coefficients(vector)
    skew = hat(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye - b * skew + c * (skew @ skew)


def so3_left_jacobian(vector: Tensor) -> Tensor:
    return so3_right_jacobian(-vector)


def so3_geodesic_distance(first: Tensor, second: Tensor) -> Tensor:
    """Shortest angular distance in radians, shape ``(...)``."""
    return torch.linalg.vector_norm(so3_log(first.transpose(-1, -2) @ second), dim=-1)


def rotation_6d_to_matrix(representation: Tensor) -> Tensor:
    """Two predicted column vectors -> SO(3) through Gram--Schmidt.

    A deterministic orthogonal fallback handles exactly zero/collinear network
    outputs; the representation is smooth away from these degenerate inputs.
    """
    first, second = representation[..., :3], representation[..., 3:]
    eps = 1e-8
    norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    e1 = torch.zeros_like(first)
    e1[..., 0] = 1
    b1 = torch.where(norm > eps, first / norm.clamp_min(eps), e1)
    residual = second - (b1 * second).sum(-1, keepdim=True) * b1
    residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    # For generic collinear vectors, roundoff can leave a residual *parallel*
    # to b1. An absolute 1e-8 cutoff misses this in float32 (e.g. all ones).
    # Scale the degeneracy test by both vector size and arithmetic precision.
    second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    tolerance = 16 * torch.finfo(representation.dtype).eps * second_norm.clamp_min(eps)
    alternate = F.one_hot(b1.abs().argmin(-1), num_classes=3).to(first)
    alternate = alternate - (alternate * b1).sum(-1, keepdim=True) * b1
    residual = torch.where(residual_norm > tolerance, residual, alternate)
    b2 = F.normalize(residual, dim=-1, eps=eps)
    b3 = F.normalize(torch.linalg.cross(b1, b2, dim=-1), dim=-1, eps=eps)
    # Remove residual projection error before returning an actual SO(3) frame.
    b2 = torch.linalg.cross(b3, b1, dim=-1)
    return torch.stack((b1, b2, b3), -1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Flatten the first two columns, inverse of rotation_6d_to_matrix."""
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def wrapped_normal_log_prob_from_tangent(
    tangent: Tensor, log_std: Tensor, num_wraps: int = 3
) -> Tensor:
    """Falorsi et al. (2018), Eq. 9, with a finite wrap sum.

    ``tangent`` is a sample from N(0, diag(exp(2 log_std))) before Exp.
    The density is with respect to the SO(3) Riemannian volume, whose uniform
    density is 1/(8 pi^2). Summing k=-num_wraps,...,+num_wraps approximates the
    infinite preimage sum; three wraps suffice for posterior std <= 1.
    """
    if num_wraps < 0:
        raise ValueError("num_wraps must be nonnegative")
    theta = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    safe_theta = theta.clamp_min(1e-8)
    # At exactly identity the wrapped density is singular. A tiny radius gives a
    # finite numerical representative; this measure-zero point is never sampled.
    axis = tangent / safe_theta
    fallback = torch.zeros_like(axis)
    fallback[..., 0] = 1
    axis = torch.where(theta > 1e-8, axis, fallback)
    wraps = torch.arange(-num_wraps, num_wraps + 1, device=tangent.device, dtype=tangent.dtype)
    radii = safe_theta + 2 * math.pi * wraps
    preimages = radii[..., :, None] * axis[..., None, :]
    standardized = preimages * torch.exp(-log_std[..., None, :])
    log_gaussian = -0.5 * standardized.square().sum(-1) - log_std.sum(-1, keepdim=True)
    log_gaussian = log_gaussian - 1.5 * math.log(2 * math.pi)
    # det J = (2 sin(theta/2)/theta)^2, with the correct limit for k=0.
    denominator = (2 * torch.sin(safe_theta / 2)).abs().clamp_min(1e-12)
    log_inverse_jacobian = 2 * (radii.abs().clamp_min(1e-12).log() - denominator.log())
    return torch.logsumexp(log_gaussian + log_inverse_jacobian, dim=-1)


def wrapped_normal_log_prob(
    rotation: Tensor, mean: Tensor, log_std: Tensor, num_wraps: int = 3
) -> Tensor:
    """Log density of ``mean @ Exp(N(0, diag(std**2)))`` (Falorsi Eq. 6)."""
    tangent = so3_log(mean.transpose(-1, -2) @ rotation)
    return wrapped_normal_log_prob_from_tangent(tangent, log_std, num_wraps)


def sample_wrapped_normal(mean: Tensor, log_std: Tensor, noise: Tensor | None = None) -> Tensor:
    """Reparameterized SO(3) posterior sample."""
    noise = torch.randn_like(log_std) if noise is None else noise
    return mean @ so3_exp(noise * log_std.exp())
