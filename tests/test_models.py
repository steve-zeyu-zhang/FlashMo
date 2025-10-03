import importlib.util

import pytest
import torch

from flashmo.geometry import so3_exp
from flashmo.models import FlashMoDiP, FrequencySparseAttention, SO3MotionVAE
from flashmo.models.block_sparse import compact_block_sparse_attention


def test_sparse_gamma_one_matches_full_attention_and_gradients():
    torch.manual_seed(2)
    sparse = FrequencySparseAttention(32, 4, beta=.5, gamma=1, backend="sdpa").double()
    reference = torch.nn.MultiheadAttention(32, 4, batch_first=True).double()
    reference.load_state_dict(sparse.state_dict())
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    actual = sparse(x, mask)
    expected = reference(x, x, x, key_padding_mask=~mask, need_weights=False)[0] * mask[..., None]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert sparse.last_backend == "sdpa"


def test_zipvl_minimum_mass_selection():
    attention = FrequencySparseAttention(4, 1, beta=1, gamma=.9, backend="sdpa")
    q = torch.ones(1, 1, 4, 4)
    k = torch.zeros_like(q)
    k[:, :, 2] = 10
    valid = torch.ones(1, 4, dtype=torch.bool)
    indices, selected_valid = attention._select(q, k, valid)
    assert attention.last_selection["kappa"].item() == 1
    assert indices.item() == 2
    assert selected_valid.item()
    k.zero_()
    attention._select(q, k, valid)
    assert attention.last_selection["kappa"].item() == 4


def test_so3_vae_reconstruction_backward_and_temporal_masks():
    torch.manual_seed(3)
    vae = SO3MotionVAE(num_joints=3, latent_joints=2, hidden_dim=32, num_layers=1,
                       num_heads=4, ff_size=64, compression=4, dropout=0)
    rotations = so3_exp(torch.randn(2, 9, 3, 3) * .4)
    translations = torch.randn(2, 9, 3) * .1
    mask = torch.arange(9)[None] < torch.tensor([9, 5])[:, None]
    output = vae(rotations, translations, mask)
    posterior = output["posterior"]
    assert posterior["rotations"].shape == (2, 3, 2, 3, 3)
    assert posterior["mask"].tolist() == [[True, True, True], [True, True, False]]
    identity = torch.eye(3).expand_as(posterior["rotations"])
    torch.testing.assert_close(posterior["rotations"].transpose(-1, -2) @ posterior["rotations"], identity,
                               atol=2e-6, rtol=2e-6)
    losses = vae.loss(output, rotations, translations, mask)
    losses["loss"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(torch.isfinite(p.grad).all() for p in vae.parameters() if p.grad is not None)
    assert vae.rotation_mean.weight.grad.abs().sum() > 0
    assert vae.rotation_log_std.weight.grad.abs().sum() > 0


def test_vae_padding_cannot_change_valid_encoding():
    torch.manual_seed(4)
    vae = SO3MotionVAE(num_joints=3, latent_joints=2, hidden_dim=32, num_layers=1,
                       num_heads=4, ff_size=64, dropout=0).eval()
    rotations = so3_exp(torch.randn(1, 8, 3, 3))
    translations = torch.randn(1, 8, 3)
    mask = torch.arange(8)[None] < 5
    first = vae.encode(rotations, translations, mask, sample=False)
    rotations[:, 5:] = so3_exp(torch.randn(1, 3, 3, 3) * 2)
    translations[:, 5:] += 100
    second = vae.encode(rotations, translations, mask, sample=False)
    torch.testing.assert_close(first["rotations"], second["rotations"])
    torch.testing.assert_close(first["translations"], second["translations"])


def test_dip_prefix_conditioning_mask_and_transfer_report():
    torch.manual_seed(5)
    model = FlashMoDiP(latent_joints=2, hidden_dim=32, num_layers=1, num_heads=4,
                       ff_size=64, text_dim=8, dropout=0, cond_mask_prob=0).eval()
    rotations = so3_exp(torch.randn(2, 4, 2, 3))
    translations = torch.randn(2, 4, 3)
    text = torch.randn(2, 3, 8)
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    kwargs = dict(text_features=text, mask=mask, prefix_rotations=rotations[:, :1],
                  prefix_translations=translations[:, :1])
    first = model(rotations, translations, torch.ones(2) * .5, **kwargs)
    assert first["rotation_velocity"].shape == (2, 4, 2, 3)
    assert not first["rotation_velocity"][1, 2:].any()
    kwargs["prefix_translations"] = kwargs["prefix_translations"] + 5
    second = model(rotations, translations, torch.ones(2) * .5, **kwargs)
    assert not torch.allclose(first["rotation_velocity"], second["rotation_velocity"])
    key = "seqTransDecoder.layers.0.self_attn.in_proj_weight"
    report = model.load_dip_checkpoint({key: torch.zeros_like(model.state_dict()[key]),
                                        "input_process.poseEmbedding.weight": torch.zeros(32, 263)})
    assert report["loaded"] == [key]
    assert "input_process.poseEmbedding.weight" in report["mismatched"]
    assert "rotation_output.weight" in report["missing"]


def test_missing_text_features_raise():
    model = FlashMoDiP(latent_joints=2, hidden_dim=16, num_layers=1, num_heads=2, text_dim=8)
    rotations = so3_exp(torch.zeros(1, 2, 2, 3))
    translations = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="frozen text_features"):
        model(rotations, translations, torch.ones(1))


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("block_sparse_attn") is None,
                    reason="Optional Block-Sparse-Attention CUDA extension unavailable")
def test_real_block_sparse_kernel_matches_sdpa_forward_backward():
    torch.manual_seed(6)
    # Non-multiple lengths and per-head kappa exercise packing and mask handling.
    q = torch.randn(2, 2, 19, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 2, 11, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    query_valid = torch.arange(19, device="cuda")[None, None] < torch.tensor([19, 13], device="cuda")[:, None, None]
    key_valid = torch.arange(11, device="cuda")[None, None] < torch.tensor([[11, 7], [5, 9]], device="cuda")[..., None]
    actual = compact_block_sparse_attention(q, k, v, query_valid, key_valid)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_valid[:, :, None])
    expected = expected * query_valid[..., None]
    torch.testing.assert_close(actual, expected, atol=.025, rtol=.025)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=.04, rtol=.04)
