import math

import pytest
import torch
from scipy.spatial.transform import Rotation

from flashmo.geometry import (
    hat, matrix_to_rotation_6d, rotation_6d_to_matrix, so3_exp,
    so3_geodesic_distance, so3_log, so3_right_jacobian, vee,
    wrapped_normal_log_prob, wrapped_normal_log_prob_from_tangent,
)


def assert_rotations(rotations, atol=1e-10):
    eye = torch.eye(3, dtype=rotations.dtype).expand_as(rotations)
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, eye, atol=atol, rtol=0)
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones_like(rotations[..., 0, 0]), atol=atol, rtol=0)


def test_exp_log_match_scipy_at_identity_generic_and_pi():
    axis = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    axis /= axis.norm()
    angles = torch.tensor([0, 1e-10, 1e-5, 0.5, 2.8, math.pi - 1e-8, math.pi], dtype=torch.float64)
    vectors = angles[:, None] * axis
    rotations = so3_exp(vectors)
    expected = torch.from_numpy(Rotation.from_rotvec(vectors.numpy()).as_matrix())
    torch.testing.assert_close(rotations, expected, atol=1e-12, rtol=1e-12)
    assert_rotations(rotations)
    torch.testing.assert_close(so3_exp(so3_log(rotations)), rotations, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(so3_log(rotations[:-1]), vectors[:-1], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("angle", [0, 1e-7, 0.4, math.pi - 1e-4])
def test_exp_log_have_finite_correct_gradients(angle):
    vector = torch.tensor([angle, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(so3_exp, (vector,), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda v: so3_log(so3_exp(v)), (vector,), eps=1e-6, atol=1e-5)


def test_right_jacobian_matches_noncommuting_matrix_derivative():
    a = torch.tensor([0.7, -0.5, 1.1], dtype=torch.float64)
    da = torch.tensor([0.4, 1.2, -0.3], dtype=torch.float64)
    dt = 1e-6
    derivative = (so3_exp(a + dt * da) - so3_exp(a - dt * da)) / (2 * dt)
    observed = vee(so3_exp(a).T @ derivative)
    expected = so3_right_jacobian(a) @ da
    torch.testing.assert_close(observed, expected, atol=2e-10, rtol=1e-9)
    assert not torch.allclose(expected, da, atol=0.1), "noncommutative correction cannot be omitted"


def test_six_dimensional_round_trip_and_degenerate_inputs():
    torch.manual_seed(17)
    rotations = so3_exp(torch.randn(30, 3, dtype=torch.float64))
    torch.testing.assert_close(rotation_6d_to_matrix(matrix_to_rotation_6d(rotations)), rotations, atol=1e-12, rtol=1e-12)
    degenerate = torch.tensor([[0.0] * 6, [1, 0, 0, 2, 0, 0]], dtype=torch.float64)
    assert_rotations(rotation_6d_to_matrix(degenerate))


def test_wrapped_density_has_correct_haar_normalization():
    # Numerical radial integration over the principal axis-angle ball; dV =
    # 4 sin²(theta/2)/theta² d³omega, sphere area is 4*pi.
    theta = torch.linspace(1e-5, math.pi, 20000, dtype=torch.float64)
    tangent = torch.stack((theta, torch.zeros_like(theta), torch.zeros_like(theta)), -1)
    log_std = torch.full_like(tangent, math.log(0.7))
    log_density = wrapped_normal_log_prob_from_tangent(tangent, log_std)
    radial_volume = 16 * math.pi * torch.sin(theta / 2).square()
    integral = torch.trapezoid(log_density.exp() * radial_volume, theta)
    torch.testing.assert_close(integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-6, rtol=0)


def test_density_is_left_invariant_and_gradients_finite():
    torch.manual_seed(19)
    mean = so3_exp(torch.randn(7, 3, dtype=torch.float64))
    log_std = torch.full((7, 3), -1.2, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn_like(log_std) * log_std.exp()
    rotation = mean @ so3_exp(tangent)
    actual = wrapped_normal_log_prob(rotation, mean, log_std)
    expected = wrapped_normal_log_prob_from_tangent(tangent, log_std)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    actual.mean().backward()
    assert torch.isfinite(log_std.grad).all()


def test_geodesic_distance_has_correct_scale_and_zero_gradient():
    eye = torch.eye(3, dtype=torch.float64)
    vector = torch.tensor([0.0, 0.0, 0.7], dtype=torch.float64, requires_grad=True)
    distance = so3_geodesic_distance(eye, so3_exp(vector))
    torch.testing.assert_close(distance, torch.tensor(0.7, dtype=torch.float64))
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    so3_geodesic_distance(eye, so3_exp(zero)).square().backward()
    assert torch.isfinite(zero.grad).all()


def test_float32_small_angles_and_group_loss_backward_are_stable():
    # Float32 cos subtraction loses precision well before float64 does. Compare
    # the entire small-angle region to the high-precision Jacobian implementation.
    angles = torch.logspace(-8, -1, 100)
    vectors = angles[:, None] * torch.tensor([[0.2, -0.3, 0.8]])
    actual = so3_right_jacobian(vectors)
    expected = so3_right_jacobian(vectors.double()).float()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    near_pi = torch.tensor([[math.pi - 1e-5, 0, 0]])
    vector = torch.cat((torch.zeros(1, 3), vectors, near_pi)).requires_grad_()
    rotation = so3_exp(vector)
    loss = so3_geodesic_distance(torch.eye(3), rotation).square().mean()
    loss += so3_log(rotation).square().mean()
    loss.backward()
    assert torch.isfinite(vector.grad).all()
    assert_rotations(rotation.detach(), atol=5e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generic_collinear_six_dimensional_frames_stay_on_so3(dtype):
    representation = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1.0000001, 1, 1],
        [0.1, 0.2, 0.3, 0.2, 0.4, 0.6],
        [1e4, -1e4, 2e4, 1e4, -1e4, 2e4],
        [1, 0, 0, 1, 1e-4, 0],
    ], dtype=dtype, requires_grad=True)
    rotations = rotation_6d_to_matrix(representation)
    assert_rotations(rotations.detach(), atol=5e-7 if dtype == torch.float32 else 1e-12)
    weights = torch.arange(9, dtype=dtype).reshape(3, 3)
    (rotations * weights).sum().backward()
    assert torch.isfinite(representation.grad).all()
    assert representation.grad[-1].abs().sum() > 0
