"""Tests for the hybrid multi-head attention module."""

from __future__ import annotations

import pytest
import torch

from deepseek_latent_attention.src.core.config import AttnConfig
from deepseek_latent_attention.src.core.hybrid_attention import MultiheadAttn


@pytest.mark.parametrize(
    "batch, seq_len, d_model, n_heads",
    [(2, 5, 16, 4), (1, 3, 8, 2)],
)
def test_multihead_attn_output_shape(
    batch: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
) -> None:
    config = AttnConfig(d_model=d_model, n_heads=n_heads)
    attn = MultiheadAttn(config)
    x = torch.randn(batch, seq_len, d_model)

    out, weights = attn(x, need_weights=True)

    assert out.shape == (batch, seq_len, d_model)
    assert weights is not None
    assert weights.shape == (batch, seq_len, seq_len)


def test_multihead_attn_applies_causal_mask() -> None:
    config = AttnConfig(d_model=8, n_heads=2, causal=True)
    attn = MultiheadAttn(config)
    attn.eval()

    x = torch.randn(1, 4, 8)
    _, weights = attn(x, need_weights=True)
    assert weights is not None

    future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    future_values = weights[0][future_mask]
    assert torch.allclose(future_values, torch.zeros_like(future_values))


def test_multihead_attn_supports_custom_mask() -> None:
    config = AttnConfig(d_model=12, n_heads=3)
    attn = MultiheadAttn(config)
    attn.eval()

    seq_len = 3
    x = torch.randn(2, seq_len, 12)
    identity_mask = torch.eye(seq_len, dtype=torch.bool)
    identity_mask = identity_mask.unsqueeze(0).unsqueeze(0)

    _, weights = attn(x, mask=identity_mask, need_weights=True)
    assert weights is not None

    expected = torch.eye(seq_len).unsqueeze(0)
    assert torch.allclose(weights, expected)
