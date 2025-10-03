"""SiT-style flow matching on a product of rotation groups and root positions.

The coupling-plan / transport / sampler organization follows willisma/SiT.
The geometric path is FlashMo section 3.2, differentiated through Exp with its
right Jacobian. See docs/MATH.md for conventions and implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from flashmo.geometry import so3_exp, so3_log, so3_right_jacobian


@dataclass
class InterpolantSample:
    rotations: Tensor
    tangent: Tensor
    velocity: Tensor
    noise: Tensor
    tangent_velocity: Tensor


class GeometricInterpolant:
    """R(t)=Exp(alpha_T(t) alpha_S(t) Log(R1) + sigma(t) epsilon).

    Time runs from noise (0) to data (1), matching official SiT. Default factors
    alpha_T=alpha_S=t give a product t**2, never silently a single t.
    Individual positive powers may be supplied per temporal/rotation factor.
    ``noise_schedule='paper_sqrt'`` uses sqrt(1-t), the time-reversed appendix
    choice; its endpoint velocity is singular and training excludes t=1.
    """

    def __init__(
        self,
        temporal_power: float | Sequence[float] = 1.0,
        spatial_power: float | Sequence[float] = 1.0,
        noise_schedule: str = "linear",
        noise_scale: float = 1.0,
    ):
        if noise_schedule not in {"linear", "paper_sqrt", "gvp"}:
            raise ValueError("noise_schedule must be linear, paper_sqrt or gvp")
        if noise_scale <= 0:
            raise ValueError("noise_scale must be positive")
        self.temporal_power = temporal_power
        self.spatial_power = spatial_power
        self.noise_schedule = noise_schedule
        self.noise_scale = float(noise_scale)

    @staticmethod
    def _time(t: Tensor | float, reference: Tensor) -> Tensor:
        t = torch.as_tensor(t, device=reference.device, dtype=reference.dtype)
        if t.ndim == 0:
            t = t.expand(reference.shape[0])
        if t.shape != (reference.shape[0],):
            raise ValueError("t must be scalar or one time per batch element")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("interpolant times must lie in [0, 1]")
        return t

    @staticmethod
    def _powers(value: float | Sequence[float], length: int, ref: Tensor) -> Tensor:
        power = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
        if power.ndim == 0:
            power = power.expand(length)
        if power.shape != (length,) or bool((power <= 0).any()):
            raise ValueError("schedule powers must be positive scalars or match the axis length")
        return power

    def coefficients(self, t: Tensor, temporal_length: int, spatial_length: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return alpha, alpha_dot, sigma, sigma_dot broadcastable to [B,L,K,3]."""
        p_t = self._powers(self.temporal_power, temporal_length, t)
        p_s = self._powers(self.spatial_power, spatial_length, t)
        power = p_t[:, None] + p_s[None, :]
        if bool((power < 1).any()):
            raise ValueError("combined temporal + spatial power must be >= 1 for finite t=0 velocity")
        time = t[:, None, None]
        if self.noise_schedule == "gvp":
            base = torch.sin(torch.pi * time / 2)
            base_derivative = (torch.pi / 2) * torch.cos(torch.pi * time / 2)
            sigma = torch.cos(torch.pi * time / 2)
            sigma_derivative = -(torch.pi / 2) * torch.sin(torch.pi * time / 2)
        else:
            base, base_derivative = time, torch.ones_like(time)
            if self.noise_schedule == "paper_sqrt":
                sigma = (1 - time).sqrt()
                sigma_derivative = -0.5 / sigma.clamp_min(1e-8)
            else:
                sigma, sigma_derivative = 1 - time, -torch.ones_like(time)
        alpha = base.pow(power)
        alpha_derivative = power * base.pow(power - 1) * base_derivative
        return (alpha[..., None], alpha_derivative[..., None],
                sigma[..., None] * self.noise_scale,
                sigma_derivative[..., None] * self.noise_scale)

    def sample(self, target: Tensor, t: Tensor | float, noise: Tensor | None = None) -> InterpolantSample:
        if target.ndim != 5 or target.shape[-2:] != (3, 3):
            raise ValueError("rotation targets must have shape [B,L,K,3,3]")
        tangent_target = so3_log(target)
        t = self._time(t, target)
        if self.noise_schedule == "paper_sqrt" and bool((t == 1).any()):
            raise ValueError("paper_sqrt has an infinite endpoint velocity; use t < 1")
        noise = torch.randn_like(tangent_target) if noise is None else noise
        if noise.shape != tangent_target.shape:
            raise ValueError("noise must have shape [B,L,K,3]")
        alpha, alpha_dot, sigma, sigma_dot = self.coefficients(t, target.shape[1], target.shape[2])
        tangent = alpha * tangent_target + sigma * noise
        tangent_velocity = alpha_dot * tangent_target + sigma_dot * noise
        velocity = (so3_right_jacobian(tangent) @ tangent_velocity[..., None]).squeeze(-1)
        return InterpolantSample(so3_exp(tangent), tangent, velocity, noise, tangent_velocity)

    def sample_euclidean(self, target: Tensor, t: Tensor | float, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Root translation uses the temporal factor and a unit spatial exponent.

        It is an ordinary R^3 position, never a rotation or HumanML3D velocity.
        When spatial_power is scalar, the same exponent applies to translation.
        """
        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError("translation targets must have shape [B,L,3]")
        t = self._time(t, target)
        noise = torch.randn_like(target) if noise is None else noise
        if noise.shape != target.shape:
            raise ValueError("translation noise must match target shape")
        # A root vector has one spatial slot. For per-joint schedules its power
        # is that of joint zero (the root), preserving the designated root rate.
        spatial = torch.as_tensor(self.spatial_power).reshape(-1)[0].item()
        root_plan = GeometricInterpolant(self.temporal_power, spatial, self.noise_schedule, self.noise_scale)
        alpha, alpha_dot, sigma, sigma_dot = root_plan.coefficients(t, target.shape[1], 1)
        return (alpha.squeeze(-2) * target + sigma.squeeze(-2) * noise,
                alpha_dot.squeeze(-2) * target + sigma_dot.squeeze(-2) * noise)


class GeometricTransport:
    """Velocity regression and Lie-group ODE sampling, inspired by SiT Transport."""

    def __init__(self, interpolant: GeometricInterpolant | None = None,
                 rotation_weight: float = 1.0, translation_weight: float = 1.0):
        self.interpolant = interpolant or GeometricInterpolant()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    @staticmethod
    def _outputs(output) -> tuple[Tensor, Tensor]:
        if isinstance(output, dict):
            return output["rotation_velocity"], output["translation_velocity"]
        return output

    @staticmethod
    def _loss(error: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return error.square().mean()
        mask = mask.to(dtype=error.dtype, device=error.device)
        if mask.shape != error.shape[:2]:
            raise ValueError("mask must have shape [B,L] with True for valid frames")
        if not bool(mask.any()):
            raise ValueError("training requires at least one valid latent frame")
        mask = mask.reshape(*mask.shape, *([1] * (error.ndim - 2))).expand_as(error)
        return (error.square() * mask).sum() / mask.sum()

    def training_losses(
        self, model: Callable, rotations: Tensor, translations: Tensor,
        t: Tensor | None = None, mask: Tensor | None = None,
        model_kwargs: dict | None = None, rotation_noise: Tensor | None = None,
        translation_noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return scalar losses; optional times/noises make validation reproducible."""
        if t is None:
            t = torch.rand(rotations.shape[0], device=rotations.device, dtype=rotations.dtype)
            if self.interpolant.noise_schedule == "paper_sqrt":
                t = t.clamp_max(1 - 1e-5)
        t = self.interpolant._time(t, rotations)
        path = self.interpolant.sample(rotations, t, rotation_noise)
        positions, position_velocity = self.interpolant.sample_euclidean(translations, t, translation_noise)
        kwargs = dict(model_kwargs or {})
        if mask is not None:
            kwargs.setdefault("mask", mask)
        predicted_rotation, predicted_translation = self._outputs(model(path.rotations, positions, t, **kwargs))
        if predicted_rotation.shape != path.velocity.shape or predicted_translation.shape != position_velocity.shape:
            raise ValueError("model velocities have incorrect rotation or translation shape")
        rotation_loss = self._loss(predicted_rotation - path.velocity, mask)
        translation_loss = self._loss(predicted_translation - position_velocity, mask)
        return {"loss": self.rotation_weight * rotation_loss + self.translation_weight * translation_loss,
                "rotation_loss": rotation_loss, "translation_loss": translation_loss}

    @torch.no_grad()
    def sample(
        self, model: Callable, shape: tuple[int, int, int] | None = None,
        steps: int = 10, method: str = "euler", model_kwargs: dict | None = None,
        device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
        mask: Tensor | None = None, initial_rotations: Tensor | None = None,
        initial_translations: Tensor | None = None, generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Integrate dR=R hat(v)dt, dx=u dt with Exp updates (10 default NFE).

        Euler uses one model call per step; Lie midpoint uses two. Starting
        rotations are Exp(N(0, noise_scale**2 I)), matching the training path,
        rather than Haar-uniform rotations. Group updates require no projection.
        """
        if steps < 1 or method not in {"euler", "midpoint"}:
            raise ValueError("steps must be positive; method must be euler or midpoint")
        if initial_rotations is not None:
            if initial_rotations.ndim != 5 or initial_rotations.shape[-2:] != (3, 3):
                raise ValueError("initial_rotations must have shape [B,L,K,3,3]")
            shape = tuple(initial_rotations.shape[:3])
            device, dtype = initial_rotations.device, initial_rotations.dtype
        if shape is None or len(shape) != 3 or min(shape) < 1:
            raise ValueError("shape must be a positive (batch, latent_frames, rotations) tuple")
        rotations = initial_rotations
        if rotations is None:
            rotations = so3_exp(torch.randn((*shape, 3), device=device, dtype=dtype, generator=generator)
                                * self.interpolant.noise_scale)
        translations = initial_translations
        if translations is None:
            translations = torch.randn((*shape[:2], 3), device=device, dtype=dtype, generator=generator)
            translations = translations * self.interpolant.noise_scale
        if translations.shape != (*shape[:2], 3):
            raise ValueError("initial_translations must have shape [B,L,3]")
        kwargs = dict(model_kwargs or {})
        if mask is not None:
            kwargs.setdefault("mask", mask)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((shape[0],), step * dt, device=device, dtype=dtype)
            omega, velocity = self._outputs(model(rotations, translations, t, **kwargs))
            if method == "midpoint":
                midpoint_rotations = rotations @ so3_exp(omega * (dt / 2))
                midpoint_translations = translations + velocity * (dt / 2)
                omega, velocity = self._outputs(model(midpoint_rotations, midpoint_translations, t + dt / 2, **kwargs))
            if mask is not None:
                omega = omega * mask[..., None, None].to(omega)
                velocity = velocity * mask[..., None].to(velocity)
            rotations = rotations @ so3_exp(dt * omega)
            translations = translations + dt * velocity
        return rotations, translations
