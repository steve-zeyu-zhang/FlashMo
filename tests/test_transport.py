import pytest
import torch

from flashmo.geometry import so3_exp, so3_log, vee
from flashmo.transport import GeometricInterpolant, GeometricTransport


@pytest.mark.parametrize("schedule", ["linear", "paper_sqrt", "gvp"])
def test_body_velocity_matches_path_finite_difference(schedule):
    torch.manual_seed(9)
    target = so3_exp(torch.randn(3, 4, 2, 3, dtype=torch.float64) * 0.6)
    noise = torch.randn(3, 4, 2, 3, dtype=torch.float64)
    plan = GeometricInterpolant(temporal_power=[1, 1.2, 1.4, 1.6], spatial_power=[1, 1.3], noise_schedule=schedule)
    t = torch.tensor([0.08, 0.4, 0.81], dtype=torch.float64)
    path = plan.sample(target, t, noise)
    dt = 1e-6
    derivative = (plan.sample(target, t + dt, noise).rotations - plan.sample(target, t - dt, noise).rotations) / (2 * dt)
    measured = vee(path.rotations.transpose(-1, -2) @ derivative)
    torch.testing.assert_close(measured, path.velocity, atol=5e-9, rtol=5e-8)


def test_default_factorization_and_endpoints():
    torch.manual_seed(4)
    target = so3_exp(torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.4)
    noise = torch.randn(2, 3, 4, 3, dtype=torch.float64)
    plan = GeometricInterpolant()
    torch.testing.assert_close(plan.sample(target, 0.0, noise).rotations, so3_exp(noise))
    torch.testing.assert_close(plan.sample(target, 1.0, noise).rotations, target)
    path = plan.sample(target, 0.5, noise)
    torch.testing.assert_close(path.tangent, 0.25 * so3_log(target) + 0.5 * noise)
    start = plan.sample(target, 0.0, noise)
    dt = 1e-7
    derivative = (plan.sample(target, dt, noise).rotations - start.rotations) / dt
    torch.testing.assert_close(vee(start.rotations.transpose(-1, -2) @ derivative), start.velocity, atol=3e-7, rtol=1e-6)


def test_euclidean_derivative_matches_finite_difference():
    torch.manual_seed(3)
    target = torch.randn(2, 5, 3, dtype=torch.float64)
    noise = torch.randn_like(target)
    t = torch.tensor([0.3, 0.6], dtype=torch.float64)
    plan = GeometricInterpolant(temporal_power=1.2, spatial_power=[1.4, 1.8])
    position, velocity = plan.sample_euclidean(target, t, noise)
    dt = 1e-6
    measured = (plan.sample_euclidean(target, t + dt, noise)[0] - plan.sample_euclidean(target, t - dt, noise)[0]) / (2 * dt)
    torch.testing.assert_close(measured, velocity, atol=1e-9, rtol=1e-8)


def test_oracle_loss_and_mask_ignore_padding():
    torch.manual_seed(8)
    target = so3_exp(torch.randn(2, 4, 3, 3, dtype=torch.float64) * 0.3)
    translations = torch.randn(2, 4, 3, dtype=torch.float64)
    noise_r = torch.randn(2, 4, 3, 3, dtype=torch.float64)
    noise_x = torch.randn_like(translations)
    t = torch.tensor([0.2, 0.6], dtype=torch.float64)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    transport = GeometricTransport()
    target_r = transport.interpolant.sample(target, t, noise_r).velocity
    target_x = transport.interpolant.sample_euclidean(translations, t, noise_x)[1]

    def oracle(r, x, time, **kwargs):
        assert torch.equal(kwargs["mask"], mask)
        return {"rotation_velocity": target_r + (~mask)[..., None, None] * 100,
                "translation_velocity": target_x + (~mask)[..., None] * 100}

    loss = transport.training_losses(oracle, target, translations, t=t, mask=mask,
                                     rotation_noise=noise_r, translation_noise=noise_x)
    torch.testing.assert_close(loss["loss"], torch.tensor(0.0, dtype=torch.float64))


@pytest.mark.parametrize("method", ["euler", "midpoint"])
def test_lie_sampler_constant_velocity_stays_on_group(method):
    torch.manual_seed(11)
    start = so3_exp(torch.randn(2, 3, 4, 3, dtype=torch.float64))
    positions = torch.randn(2, 3, 3, dtype=torch.float64)
    omega = torch.randn(2, 3, 4, 3, dtype=torch.float64)
    speed = torch.randn_like(positions)
    count = 0

    def field(r, x, t):
        nonlocal count
        count += 1
        assert not torch.is_grad_enabled()
        return {"rotation_velocity": omega, "translation_velocity": speed}

    rotations, translations = GeometricTransport().sample(field, steps=10, method=method,
        initial_rotations=start, initial_translations=positions)
    torch.testing.assert_close(rotations, start @ so3_exp(omega), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(translations, positions + speed, atol=1e-12, rtol=1e-12)
    eye = torch.eye(3, dtype=start.dtype).expand_as(start)
    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, eye, atol=1e-12, rtol=1e-12)
    assert count == 10 * (2 if method == "midpoint" else 1)


def test_lie_midpoint_improves_noncommutative_time_varying_field():
    # This field rotates about non-parallel axes over time; a single Exp of the
    # integrated vector is not an exact solution. Compare to a refined solver.
    start = torch.eye(3, dtype=torch.float64).reshape(1, 1, 1, 3, 3)
    zero = torch.zeros(1, 1, 3, dtype=torch.float64)
    def field(r, x, t):
        omega = torch.stack((1 + t, 2 * t, 1 - t), -1)[:, None, None]
        return omega, torch.zeros_like(x)
    transport = GeometricTransport()
    kwargs = dict(initial_rotations=start, initial_translations=zero)
    reference = transport.sample(field, steps=1000, method="midpoint", **kwargs)[0]
    euler = transport.sample(field, steps=10, method="euler", **kwargs)[0]
    midpoint = transport.sample(field, steps=10, method="midpoint", **kwargs)[0]
    assert (midpoint - reference).norm() < (euler - reference).norm() / 10


def test_invalid_schedules_and_empty_loss_masks_fail():
    rotations = torch.eye(3).expand(1, 2, 3, 3, 3)
    with pytest.raises(ValueError, match="infinite endpoint"):
        GeometricInterpolant(noise_schedule="paper_sqrt").sample(rotations, 1.0)
    with pytest.raises(ValueError, match="combined"):
        GeometricInterpolant(0.1, 0.2).sample(rotations, 0.3)
    with pytest.raises(ValueError, match="at least one"):
        GeometricTransport._loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))
