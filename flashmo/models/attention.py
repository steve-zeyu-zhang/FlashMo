"""FlashMo §3.3 structured low-frequency key/value selection.

Selection follows ZipVL (arXiv:2410.08584v2), Equations 6--9, using
nonnegative softmax mass and normalization by nonzero column counts.
The fixed Lo-Fi/Hi-Fi head split follows LITv2's HiLo concept; FlashMo replaces
its image pooling/local windows with adaptive temporal K/V/full attention.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .block_sparse import availability, compact_block_sparse_attention


class FrequencySparseAttention(nn.Module):
    """Keep every query, physically gather selected Lo-Fi keys and values.

    As in the paper, the first ``int(beta * num_heads)`` heads are Lo-Fi.
    Selection is identical during training and inference. Scoring is quadratic:
    this reference implementation does not claim the paper's measured speedup.
    """

    def __init__(self, embed_dim, num_heads, beta=0.5, gamma=0.95,
                 dropout=0.0, selection_chunk_size=64, backend="auto", selection_queries=0,
                 attention_dtype=None):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0 <= beta <= 1 or not 0 < gamma <= 1:
            raise ValueError("beta must be in [0,1] and gamma in (0,1]")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.num_lo_heads = int(beta * num_heads)
        self.gamma, self.dropout = gamma, dropout
        self.selection_chunk_size = selection_chunk_size
        if backend not in {"auto", "sdpa", "block_sparse"}:
            raise ValueError("backend must be auto, sdpa, or block_sparse")
        self.backend, self.selection_queries = backend, selection_queries
        if attention_dtype not in (None, "bfloat16", "float16"):
            raise ValueError("attention_dtype must be None, bfloat16, or float16")
        self.attention_dtype = attention_dtype
        self.last_backend, self.backend_reason = None, None
        # Parameter names deliberately match torch.nn.MultiheadAttention / DiP.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)
        self.last_selection = None

    @torch.no_grad()
    def _select(self, q, k, valid):
        batch, heads, length, _ = q.shape
        mass = torch.zeros(batch, heads, length, device=q.device, dtype=torch.float32)
        counts = torch.zeros_like(mass)
        query_valid = valid
        if self.selection_queries and length > self.selection_queries:
            # ZipVL's fixed query budget, uniformly spaced over each valid clip.
            query_indices = []
            for row in valid:
                positions = row.nonzero().squeeze(-1)
                slots = torch.linspace(0, len(positions) - 1, self.selection_queries,
                                       device=q.device).round().long()
                query_indices.append(positions[slots])
            query_indices = torch.stack(query_indices)
            q = q.gather(-2, query_indices[:, None, :, None].expand(-1, heads, -1, self.head_dim))
            query_valid = torch.ones(batch, self.selection_queries, dtype=torch.bool, device=q.device)
        query_length = q.shape[-2]
        for start in range(0, query_length, self.selection_chunk_size):
            end = min(start + self.selection_chunk_size, query_length)
            score = (q[:, :, start:end].float() @ k.float().transpose(-1, -2))
            score *= self.head_dim ** -0.5
            score.masked_fill_(~valid[:, None, None, :], -torch.inf)
            prob = score.softmax(-1) * query_valid[:, None, start:end, None]
            mass += prob.sum(-2)
            counts += (prob != 0).sum(-2)
        sorted_mass = mass.sort(dim=-1, descending=True).values
        target = self.gamma * sorted_mass.sum(-1, keepdim=True)
        kappa = (sorted_mass.cumsum(-1) < target).sum(-1) + 1
        valid_counts = valid.sum(-1)[:, None].expand(-1, heads)
        kappa = torch.minimum(kappa, valid_counts).clamp_min(1)
        if self.gamma == 1:
            kappa = valid_counts
        normalized = mass / counts.clamp_min(1)
        normalized.masked_fill_(~valid[:, None, :], -torch.inf)
        # Padding to max kappa permits standard fused SDPA; padded K/V masked out.
        retained = int(kappa.max().item())
        indices = normalized.topk(retained, dim=-1).indices
        selected_valid = torch.arange(retained, device=q.device)[None, None] < kappa[..., None]
        self.last_selection = {"kappa": kappa.detach(), "indices": indices.detach(),
                               "valid": selected_valid.detach(), "tokens": length}
        return indices, selected_valid

    def _attend(self, q, k, v, query_valid, key_valid, dropout):
        original_dtype = q.dtype
        kernel_dtype = self.attention_dtype or ("bfloat16" if self.backend == "block_sparse" else None)
        if self.backend != "sdpa" and q.is_cuda and q.dtype == torch.float32 and kernel_dtype:
            dtype = getattr(torch, kernel_dtype)
            kernel_q, kernel_k, kernel_v = q.to(dtype), k.to(dtype), v.to(dtype)
        else:
            kernel_q, kernel_k, kernel_v = q, k, v
        usable, reason = availability(kernel_q) if self.backend != "sdpa" else (False, "explicit SDPA")
        if usable:
            self.last_backend, self.backend_reason = "block_sparse", reason
            return compact_block_sparse_attention(kernel_q, kernel_k, kernel_v,
                                                   query_valid, key_valid, dropout).to(original_dtype)
        if self.backend == "block_sparse":
            raise RuntimeError(f"Block-Sparse-Attention requested but {reason}")
        self.last_backend, self.backend_reason = "sdpa", reason
        return F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, :, None], dropout_p=dropout)

    def forward(self, x, mask=None):
        batch, length, width = x.shape
        if mask is None:
            mask = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        if mask.shape != (batch, length) or not mask.any(-1).all():
            raise ValueError("Each sequence needs at least one valid attention token")
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        dropout = self.dropout if self.training else 0.0
        pieces = []
        if self.num_lo_heads:
            lo = self.num_lo_heads
            indices, selected_valid = self._select(q[:, :lo], k[:, :lo], mask)
            gather = indices[..., None].expand(-1, -1, -1, self.head_dim)
            keys = k[:, :lo].gather(-2, gather)
            values = v[:, :lo].gather(-2, gather)
            pieces.append(self._attend(q[:, :lo], keys, values, mask[:, None], selected_valid, dropout))
        if self.num_lo_heads < self.num_heads:
            pieces.append(self._attend(
                q[:, self.num_lo_heads:], k[:, self.num_lo_heads:], v[:, self.num_lo_heads:],
                mask[:, None], mask[:, None], dropout))
        attended = torch.cat(pieces, dim=1).transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(attended) * mask[..., None]

    @staticmethod
    @torch.no_grad()
    def frequency_energy(head_features):
        """Diagnostic FFT power [B,H,F] from [B,H,T,D], no learned partition claim."""
        return torch.fft.rfft(head_features.float(), dim=-2).abs().square().mean(-1)
