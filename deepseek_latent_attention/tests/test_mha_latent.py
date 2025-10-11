"""Unit tests for LatentAttention modules."""
from __future__ import annotations

import pytest

try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled during CI
    pytest.skip("PyTorch is required for latent attention tests", allow_module_level=True)

from deepseek_latent_attention.src.core.mha_latent import LatentAttention, LatentSparseAttention


def test_latent_equals_dense_when_ratio_one() -> None:
    torch.manual_seed(0)
    embed_dim = 8
    num_heads = 2
    seq_len = 4
    batch = 2
    latent = LatentAttention(embed_dim, num_heads, latent_dim_ratio=1.0, dropout=0.0, use_bias=False)
    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=False, batch_first=True)

    eye = torch.eye(embed_dim)
    latent.q_proj.weight.data.copy_(eye)
    latent.k_proj.weight.data.copy_(eye)
    latent.v_proj.weight.data.copy_(eye)
    latent.out_proj.weight.data.copy_(eye)
    head_dim = embed_dim // num_heads
    latent.q_latent.weight.data.copy_(torch.eye(head_dim))
    latent.k_latent.weight.data.copy_(torch.eye(head_dim))

    mha.in_proj_weight.data.zero_()
    mha.in_proj_weight.data[:embed_dim] = eye
    mha.in_proj_weight.data[embed_dim : 2 * embed_dim] = eye
    mha.in_proj_weight.data[2 * embed_dim :] = eye
    mha.out_proj.weight.data.copy_(eye)

    inputs = torch.randn(batch, seq_len, embed_dim)
    latent_out, _, _ = latent(inputs, inputs, inputs, need_weights=True)
    mha_out, _ = mha(inputs, inputs, inputs, need_weights=True)
    assert torch.allclose(latent_out, mha_out, atol=1e-6)


def test_gradient_flow() -> None:
    torch.manual_seed(0)
    embed_dim = 16
    num_heads = 4
    seq_len = 5
    batch = 3
    latent = LatentAttention(embed_dim, num_heads)
    inputs = torch.randn(batch, seq_len, embed_dim, requires_grad=True)
    out, _, _ = latent(inputs, inputs, inputs)
    loss = out.pow(2).mean()
    loss.backward()
    assert inputs.grad is not None
    assert not torch.isnan(inputs.grad).any()


def test_sparse_mask_matches_dense_when_full() -> None:
    torch.manual_seed(0)
    embed_dim = 12
    num_heads = 3
    seq_len = 6
    batch = 2
    dense = LatentAttention(embed_dim, num_heads)
    sparse = LatentSparseAttention(embed_dim, num_heads)
    full_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    sparse_cfg = {"binary_mask": full_mask}

    inputs = torch.randn(batch, seq_len, embed_dim)
    dense_out, _, _ = dense(inputs, inputs, inputs)
    sparse_out, _, _ = sparse(inputs, inputs, inputs, sparse_config=sparse_cfg)
    assert torch.allclose(dense_out, sparse_out, atol=1e-6)


def test_parameter_count_reduced_by_latent_ratio() -> None:
    embed_dim = 32
    num_heads = 4
    latent_small = LatentAttention(embed_dim, num_heads, latent_dim_ratio=0.25)
    latent_large = LatentAttention(embed_dim, num_heads, latent_dim_ratio=1.0)
    params_small = sum(p.numel() for p in latent_small.parameters())
    params_large = sum(p.numel() for p in latent_large.parameters())
    assert params_small < params_large
