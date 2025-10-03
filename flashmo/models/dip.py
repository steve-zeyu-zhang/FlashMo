"""DiP decoder adapted to FlashMo manifold velocity prediction.

Architecture adapted from GuyTevet/motion-diffusion-model/model/mdm.py,
commit ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778 (MIT, Guy Tevet 2022).
The decoder block parameter names preserve compatibility with DiP weights.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from flashmo.geometry import matrix_to_rotation_6d
from .attention import FrequencySparseAttention


def sinusoidal_embedding(values, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        half, device=values.device, dtype=torch.float32) / max(half, 1))
    angles = values.float()[..., None] * frequencies
    # Same interleaved sin/cos convention as DiP's positional lookup table.
    result = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
    return F.pad(result, (0, dim - result.shape[-1]))


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        return self.dropout(x + sinusoidal_embedding(positions, self.hidden_dim).to(x.dtype))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                        nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t):
        # Continuous t in [0,1], not an integer DDPM timestep lookup.
        return self.time_embed(sinusoidal_embedding(t * 1000, self.hidden_dim))


class InputProcess(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, rotations, translations):
        features = matrix_to_rotation_6d(rotations).flatten(-2)
        return self.poseEmbedding(torch.cat((features, translations), dim=-1))


class SparseDecoderLayer(nn.Module):
    """DiP post-norm decoder with only self-attention replaced by §3.3."""

    def __init__(self, hidden_dim, num_heads, ff_size, dropout, beta, gamma,
                 attention_backend="auto", selection_queries=0, attention_dtype=None):
        super().__init__()
        self.self_attn = FrequencySparseAttention(hidden_dim, num_heads, beta, gamma, dropout,
                                                 backend=attention_backend, selection_queries=selection_queries,
                                                 attention_dtype=attention_dtype)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                                   batch_first=True)
        self.linear1, self.linear2 = nn.Linear(hidden_dim, ff_size), nn.Linear(ff_size, hidden_dim)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(hidden_dim) for _ in range(3)]
        self.dropout, self.dropout1, self.dropout2, self.dropout3 = [nn.Dropout(dropout) for _ in range(4)]

    def forward(self, x, memory, mask, memory_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, mask)))
        cross = self.multihead_attn(x, memory, memory, key_padding_mask=~memory_mask,
                                   need_weights=False)[0]
        x = self.norm2(x + self.dropout2(cross))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.norm3(x + self.dropout3(ff)) * mask[..., None]


class DecoderStack(nn.Module):
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(SparseDecoderLayer(**kwargs) for _ in range(num_layers))

    def forward(self, x, memory, mask, memory_mask):
        for layer in self.layers:
            x = layer(x, memory, mask, memory_mask)
        return x


class FlashMoDiP(nn.Module):
    """Clean prefix + noisy suffix, text cross-attention, SO(3) body velocities.

    ``mask`` and ``text_mask`` use True for valid tokens. Cached text features
    must come from the selected frozen encoder for production training.
    """

    def __init__(self, latent_joints=22, hidden_dim=512, num_layers=8, num_heads=4,
                 ff_size=1024, text_dim=512, dropout=0.1, beta=0.5, gamma=0.95,
                 cond_mask_prob=0.1, attention_backend="auto", selection_queries=0, attention_dtype=None):
        super().__init__()
        self.latent_joints, self.hidden_dim, self.text_dim = latent_joints, hidden_dim, text_dim
        self.cond_mask_prob = cond_mask_prob
        self.input_process = InputProcess(6 * latent_joints + 3, hidden_dim)
        self.sequence_pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.embed_timestep = TimestepEmbedder(hidden_dim)
        self.embed_text = nn.Linear(text_dim, hidden_dim)
        self.seqTransDecoder = DecoderStack(num_layers, hidden_dim=hidden_dim, num_heads=num_heads,
                                            ff_size=ff_size, dropout=dropout, beta=beta, gamma=gamma,
                                            attention_backend=attention_backend, selection_queries=selection_queries,
                                            attention_dtype=attention_dtype)
        self.rotation_output = nn.Linear(hidden_dim, latent_joints * 3)
        self.translation_output = nn.Linear(hidden_dim, 3)

    def forward(self, rotations, translations, t, text_features=None, text_mask=None,
                mask=None, prefix_rotations=None, prefix_translations=None,
                prefix_mask=None, force_unconditional=False):
        batch, length, joints = rotations.shape[:3]
        if rotations.shape != (batch, length, self.latent_joints, 3, 3):
            raise ValueError("rotations must have shape [B,L,latent_joints,3,3]")
        if translations.shape != (batch, length, 3):
            raise ValueError("translations must have shape [B,L,3]")
        if t.ndim == 0:
            t = t.expand(batch)
        if t.shape != (batch,):
            raise ValueError("t must be scalar or shape [B]")
        if mask is None:
            mask = torch.ones(batch, length, dtype=torch.bool, device=rotations.device)
        suffix_mask = mask
        prefix_length = 0
        if prefix_rotations is not None:
            if prefix_translations is None:
                raise ValueError("prefix_translations is required with prefix_rotations")
            prefix_length = prefix_rotations.shape[1]
            if prefix_mask is None:
                prefix_mask = torch.ones(batch, prefix_length, dtype=torch.bool, device=rotations.device)
            rotations = torch.cat((prefix_rotations, rotations), dim=1)
            translations = torch.cat((prefix_translations, translations), dim=1)
            mask = torch.cat((prefix_mask, mask), dim=1)
        if text_features is None:
            if not force_unconditional:
                raise ValueError("Pass frozen text_features, or explicitly force_unconditional=True")
            text_features = translations.new_zeros(batch, 1, self.text_dim)
        if text_features.shape[0] != batch or text_features.shape[-1] != self.text_dim:
            raise ValueError("text_features has incompatible batch/feature dimension")
        if text_mask is None:
            text_mask = torch.ones(text_features.shape[:2], dtype=torch.bool, device=rotations.device)
        if not text_mask.any(-1).all():
            raise ValueError("Every text needs a valid token (empty prompts still have special tokens)")
        if force_unconditional:
            text_features = torch.zeros_like(text_features)
        elif self.training and self.cond_mask_prob:
            drop = torch.rand(batch, 1, 1, device=rotations.device) < self.cond_mask_prob
            text_features = text_features.masked_fill(drop, 0)
        memory = self.embed_text(text_features) + self.embed_timestep(t)[:, None]
        x = self.sequence_pos_encoder(self.input_process(rotations, translations))
        hidden = self.seqTransDecoder(x, memory, mask, text_mask)[:, prefix_length:]
        angular = self.rotation_output(hidden).reshape(batch, length, joints, 3)
        linear = self.translation_output(hidden)
        return {"rotation_velocity": angular * suffix_mask[..., None, None],
                "translation_velocity": linear * suffix_mask[..., None]}

    def forward_with_cfg(self, rotations, translations, t, guidance_scale=1.0, **kwargs):
        conditional = self(rotations, translations, t, **kwargs)
        if guidance_scale == 1:
            return conditional
        kwargs["force_unconditional"] = True
        unconditional = self(rotations, translations, t, **kwargs)
        return {key: unconditional[key] + guidance_scale * (value - unconditional[key])
                for key, value in conditional.items()}

    def load_dip_checkpoint(self, checkpoint):
        """Transfer exactly shape-compatible DiP tensors and return a full report.

        New manifold input/output projections cannot inherit 263-D HumanML3D
        projections. Missing/mismatched tensors remain freshly initialized.
        """
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for wrapper in ("model", "state_dict"):
            if wrapper in checkpoint and isinstance(checkpoint[wrapper], dict):
                checkpoint = checkpoint[wrapper]
        own = self.state_dict()
        compatible, mismatched, unexpected = {}, {}, []
        for key, value in checkpoint.items():
            name = key.removeprefix("module.")
            if name not in own:
                unexpected.append(name)
            elif own[name].shape != value.shape:
                mismatched[name] = {"source": list(value.shape), "target": list(own[name].shape)}
            else:
                compatible[name] = value
        self.load_state_dict(compatible, strict=False)
        return {"loaded": sorted(compatible), "mismatched": mismatched,
                "missing": sorted(set(own) - set(compatible)), "unexpected": sorted(unexpected)}
