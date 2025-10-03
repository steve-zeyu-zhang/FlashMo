"""Adapter for MIT HAN Lab's Block-Sparse-Attention CUDA kernel.

FlashMo selects individual K/V tokens, while this kernel masks 128-token
blocks. We therefore compact selected tokens *before* calling the kernel,
packing each batch/head as a sequence. Every block of the compacted sequence
is active. This preserves token selection exactly instead of rounding it to
whole original blocks (which would change the paper's operator).
"""

import importlib

import torch


def availability(q):
    if not q.is_cuda:
        return False, "requires CUDA"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False, "requires float16/bfloat16 (use CUDA autocast)"
    if q.shape[-1] not in (32, 64, 128):
        return False, "requires head dimension 32, 64, or 128"
    try:
        importlib.import_module("block_sparse_attn")
    except (ImportError, OSError) as error:
        return False, f"CUDA extension unavailable: {error}"
    return True, "available"


def compact_block_sparse_attention(q, k, v, query_valid, key_valid, dropout=0.0):
    """Differentiable exact attention [B,H,Q,D] x compact [B,H,K,D]."""
    from block_sparse_attn import block_sparse_attn_func

    batch, heads, query_length, width = q.shape
    key_length = k.shape[-2]
    query_valid = query_valid.expand(batch, heads, query_length).reshape(batch * heads, query_length)
    key_valid = key_valid.expand(batch, heads, key_length).reshape(batch * heads, key_length)
    qflat = q.reshape(batch * heads, query_length, width)
    kflat, vflat = k.reshape(batch * heads, key_length, width), v.reshape(batch * heads, key_length, width)
    q_lengths, k_lengths = query_valid.sum(-1), key_valid.sum(-1)
    if not (q_lengths > 0).all() or not (k_lengths > 0).all():
        raise ValueError("Block sparse attention requires nonempty packed sequences")
    cu_q = torch.cat((q_lengths.new_zeros(1), q_lengths.cumsum(0))).to(torch.int32)
    cu_k = torch.cat((k_lengths.new_zeros(1), k_lengths.cumsum(0))).to(torch.int32)
    max_q, max_k = int(q_lengths.max()), int(k_lengths.max())
    # Explicit 128x128 mask over compacted sequences. Variable lengths discard
    # final block padding in the kernel; no padded key contributes to softmax.
    block_mask = torch.ones(batch * heads, 1, math_ceil128(max_q), math_ceil128(max_k),
                            dtype=torch.bool, device=q.device)
    head_mask = torch.ones(1, dtype=torch.int32, device=q.device)
    output = block_sparse_attn_func(
        qflat[query_valid].unsqueeze(1).contiguous(),
        kflat[key_valid].unsqueeze(1).contiguous(),
        vflat[key_valid].unsqueeze(1).contiguous(),
        cu_q, cu_k, head_mask, None, block_mask, max_q, max_k,
        dropout, is_causal=False)
    flat_output = torch.zeros_like(qflat)
    flat_output[query_valid] = output.squeeze(1)
    return flat_output.reshape(batch, heads, query_length, width)


def math_ceil128(length):
    return (length + 127) // 128
