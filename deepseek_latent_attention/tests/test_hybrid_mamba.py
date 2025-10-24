"""Tests for the hybrid Mamba attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepseek_latent_attention.src.core.hybrid_mamba_attn import (
    HybridMambaAttention,
    HybridMambaConfig,
)
from deepseek_latent_attention.src.core.mamba_block import MambaBlock, MambaConfig


class DummyAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.last_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.last_mask = mask
        weights = torch.full((x.size(0), x.size(1), x.size(1)), 1.0 / x.size(1))
        if need_weights:
            return x + 1.0, weights
        return x + 1.0, None


def _build_module(bias: float) -> HybridMambaAttention:
    torch.manual_seed(0)
    base = DummyAttention(d_model=4)
    mamba = MambaBlock(
        MambaConfig(d_model=4, state_size=4, dt_rank=2, conv_kernel_size=2)
    )
    config = HybridMambaConfig(enable=True, gate_bias=bias)
    module = HybridMambaAttention(base, mamba, config=config)
    module.gate_proj.bias.data.fill_(bias)
    return module


def test_hybrid_mamba_gate_zero_matches_attention() -> None:
    module = _build_module(bias=-10.0)
    x = torch.randn(1, 3, 4)
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    tokens = [["<cot>", "middle", "end:"]]

    out, stats = module(x, mask=mask, tokens=tokens, return_stats=True)

    assert module.base_attn.last_mask is mask
    expected, _ = module.base_attn(x, mask=mask)
    assert torch.allclose(out, expected, atol=1e-4)
    assert "gate" in stats and stats["gate"].shape == x.shape


def test_hybrid_mamba_gate_one_matches_ssm() -> None:
    module = _build_module(bias=25.0)
    x = torch.randn(1, 3, 4)
    tokens = [["<cot>", "middle", "end:"]]

    ssm_out = module.mamba(x)
    fused, _ = module(x, tokens=tokens)

    assert torch.allclose(fused, ssm_out, atol=1e-4)


def test_hybrid_mamba_stats_include_anchor_indices() -> None:
    module = _build_module(bias=0.0)
    x = torch.randn(1, 4, 4)
    tokens = [["<cot>", "middle", "item:", "tail"]]

    _, stats = module(x, tokens=tokens, return_stats=True)

    assert stats["meta_token_indices"].numel() > 0
    assert stats["ssm_norm"].shape == (1, 4, 1)
