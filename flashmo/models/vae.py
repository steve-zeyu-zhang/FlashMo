"""Temporal SO(3) VAE using Falorsi's Lie-group reparameterization.

Adapted from pimdh/lie-vae (MIT, Falorsi and de Haan), with an SO(3)-valued
variational posterior and temporal motion encoding.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from flashmo.geometry import (matrix_to_rotation_6d, rotation_6d_to_matrix,
                              so3_exp, so3_geodesic_distance,
                              wrapped_normal_log_prob,
                              wrapped_normal_log_prob_from_tangent)
from .dip import PositionalEncoding


SMPL22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def forward_kinematics(rotations, translations, offsets, parents=SMPL22_PARENTS):
    """Differentiable forward kinematics using joint rest offsets.

    Rotations [B,F,J,3,3], root translations [B,F,3], offsets [J,3] or [B,J,3].
    Root translations are the pelvis trajectory, root offset is ignored.
    """
    if rotations.shape[-3] != len(parents):
        raise ValueError("Number of joints must match parents")
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(rotations.shape[0], -1, -1)
    world_rotations, positions = [], []
    for joint, parent in enumerate(parents):
        if parent < 0:
            world_rotations.append(rotations[:, :, joint])
            positions.append(translations)
        else:
            world_rotations.append(world_rotations[parent] @ rotations[:, :, joint])
            displacement = (world_rotations[parent] @ offsets[:, None, joint, :, None]).squeeze(-1)
            positions.append(positions[parent] + displacement)
    return torch.stack(positions, dim=2)


def masked_mean(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


class SO3MotionVAE(nn.Module):
    def __init__(self, num_joints=22, latent_joints=22, hidden_dim=512, num_layers=4,
                 num_heads=4, compression=4, ff_size=1024, dropout=0.1,
                 prior_type="wrapped_normal", num_wraps=3):
        super().__init__()
        if compression < 1 or prior_type not in {"wrapped_normal", "haar"}:
            raise ValueError("Invalid compression or prior_type")
        self.num_joints, self.latent_joints = num_joints, latent_joints
        self.hidden_dim, self.compression = hidden_dim, compression
        self.prior_type, self.num_wraps = prior_type, num_wraps
        self.input_projection = nn.Linear(num_joints * 6 + 3, hidden_dim)
        # Non-overlapping learned temporal compression prevents invalid padding
        # from bleeding into another valid block. Padding is zeroed explicitly.
        self.downsample = nn.Conv1d(hidden_dim, hidden_dim, compression, stride=compression)
        self.position = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, ff_size, dropout,
                                                    activation="gelu", batch_first=True)
        decoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, ff_size, dropout,
                                                    activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers, enable_nested_tensor=False)
        self.rotation_mean = nn.Linear(hidden_dim, 6 * latent_joints)
        self.rotation_log_std = nn.Linear(hidden_dim, 3 * latent_joints)
        self.translation_mean = nn.Linear(hidden_dim, 3)
        self.translation_log_std = nn.Linear(hidden_dim, 3)
        self.latent_projection = nn.Linear(latent_joints * 6 + 3, hidden_dim)
        self.upsample = nn.ConvTranspose1d(hidden_dim, hidden_dim, compression, stride=compression)
        self.rotation_decoder = nn.Linear(hidden_dim, num_joints * 6)
        self.translation_decoder = nn.Linear(hidden_dim, 3)
        # Bias near an identity 6D frame, avoiding near-zero Gram-Schmidt means.
        with torch.no_grad():
            identity6 = matrix_to_rotation_6d(torch.eye(3))
            self.rotation_mean.bias.copy_(identity6.repeat(latent_joints))
            self.rotation_decoder.bias.copy_(identity6.repeat(num_joints))
            self.rotation_log_std.bias.fill_(-2)
            self.translation_log_std.bias.fill_(-2)

    def encode(self, rotations, translations, mask=None, sample=True):
        batch, frames, joints = rotations.shape[:3]
        if rotations.shape != (batch, frames, self.num_joints, 3, 3):
            raise ValueError("rotations must be [B,F,num_joints,3,3]")
        if translations.shape != (batch, frames, 3):
            raise ValueError("translations must be [B,F,3]")
        if mask is None:
            mask = torch.ones(batch, frames, dtype=torch.bool, device=rotations.device)
        if not mask.any(-1).all():
            raise ValueError("Each motion needs at least one valid frame")
        features = torch.cat((matrix_to_rotation_6d(rotations).flatten(-2), translations), dim=-1)
        features = self.input_projection(features) * mask[..., None]
        padding = (-frames) % self.compression
        features = F.pad(features.transpose(1, 2), (0, padding))
        padded_mask = F.pad(mask, (0, padding), value=False)
        latent_mask = padded_mask.reshape(batch, -1, self.compression).any(-1)
        hidden = self.downsample(features).transpose(1, 2)
        hidden = self.encoder(self.position(hidden), src_key_padding_mask=~latent_mask)
        length = hidden.shape[1]
        means = rotation_6d_to_matrix(self.rotation_mean(hidden).reshape(batch, length, self.latent_joints, 6))
        log_std = self.rotation_log_std(hidden).reshape(batch, length, self.latent_joints, 3).clamp(-4, 0)
        translation_mean = self.translation_mean(hidden)
        translation_log_std = self.translation_log_std(hidden).clamp(-4, 1)
        # Falorsi: noise in so(3), exponential map, left translate by mean.
        tangent = torch.randn_like(log_std) * log_std.exp() if sample else torch.zeros_like(log_std)
        latent_rotation = means @ so3_exp(tangent)
        latent_translation = translation_mean
        if sample:
            latent_translation = latent_translation + torch.randn_like(translation_mean) * translation_log_std.exp()
        # KL is a reparameterized MC estimate on SO(3), including exp Jacobian
        # and wrapping; never use the Euclidean diagonal Gaussian KL for rotation.
        # Even deterministic encoding computes KL at a genuine sample.
        kl_tangent = tangent if sample else torch.randn_like(log_std) * log_std.exp()
        kl_rotation = latent_rotation if sample else means @ so3_exp(kl_tangent)
        log_q = wrapped_normal_log_prob_from_tangent(kl_tangent, log_std, self.num_wraps)
        if self.prior_type == "haar":
            log_p = -math.log(8 * math.pi ** 2)
        else:
            identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
            log_p = wrapped_normal_log_prob(kl_rotation, identity, torch.zeros_like(log_std), self.num_wraps)
        rotation_kl = (log_q - log_p).sum(-1)
        translation_kl = 0.5 * (translation_mean.square() + (2 * translation_log_std).exp()
                                - 1 - 2 * translation_log_std).sum(-1)
        return {"rotations": latent_rotation, "translations": latent_translation,
                "mask": latent_mask, "kl": rotation_kl + translation_kl,
                "rotation_kl": rotation_kl, "translation_kl": translation_kl,
                "rotation_mean": means, "rotation_log_std": log_std,
                "translation_mean": translation_mean, "translation_log_std": translation_log_std}

    def decode(self, rotations, translations, length, mask=None):
        batch, latent_length = rotations.shape[:2]
        if length < 1 or length > latent_length * self.compression:
            raise ValueError("Requested length outside decoder temporal capacity")
        if mask is None:
            mask = torch.ones(batch, latent_length, dtype=torch.bool, device=rotations.device)
        features = torch.cat((matrix_to_rotation_6d(rotations).flatten(-2), translations), dim=-1)
        hidden = self.latent_projection(features) * mask[..., None]
        hidden = self.upsample(hidden.transpose(1, 2)).transpose(1, 2)[:, :length]
        frame_mask = mask.repeat_interleave(self.compression, dim=1)[:, :length]
        hidden = self.decoder(self.position(hidden), src_key_padding_mask=~frame_mask)
        decoded_rotations = rotation_6d_to_matrix(
            self.rotation_decoder(hidden).reshape(batch, length, self.num_joints, 6))
        return {"rotations": decoded_rotations, "translations": self.translation_decoder(hidden),
                "mask": frame_mask}

    def forward(self, rotations, translations, mask=None, sample=True):
        posterior = self.encode(rotations, translations, mask=mask, sample=sample)
        output = self.decode(posterior["rotations"], posterior["translations"],
                             rotations.shape[1], mask=posterior["mask"])
        output["posterior"] = posterior
        return output

    def loss(self, output, rotations, translations, mask=None, positions=None, offsets=None,
             rotation_weight=1., translation_weight=1., position_weight=1., kl_weight=1e-4):
        if mask is None:
            mask = torch.ones(rotations.shape[:2], dtype=torch.bool, device=rotations.device)
        rotation = masked_mean(so3_geodesic_distance(output["rotations"], rotations).square(), mask)
        translation = masked_mean((output["translations"] - translations).square(), mask)
        position = translation.new_zeros(())
        if positions is not None:
            if offsets is None:
                raise ValueError("True SMPL rest offsets required for positional FK loss")
            reconstructed = forward_kinematics(output["rotations"], output["translations"], offsets)
            position = masked_mean((reconstructed - positions).square(), mask)
        posterior = output["posterior"]
        kl = masked_mean(posterior["kl"], posterior["mask"])
        total = rotation_weight * rotation + translation_weight * translation + position_weight * position + kl_weight * kl
        return {"loss": total, "rotation": rotation, "translation": translation, "position": position, "kl": kl}
