"""Tests for the Mamba anchor gate adjustments."""

from __future__ import annotations

import torch

from deepseek_latent_attention.src.core.mamba_anchor import MambaAnchor, MambaAnchorConfig


def test_mamba_anchor_adjusts_only_anchor_positions() -> None:
    gate = torch.full((1, 4, 2), 0.5)
    tokens = [["prefix", "<cot>", "body", "tail:"]]
    config = MambaAnchorConfig(enable=True, alpha=0.4)

    adjusted, mask = MambaAnchor(config)(gate, tokens=tokens)

    diff = adjusted - gate
    mask_3d = mask.unsqueeze(-1).expand_as(diff)
    stationary = torch.zeros_like(diff[~mask_3d])
    assert torch.allclose(diff[~mask_3d], stationary)
    assert torch.all(diff[mask_3d] >= 0)


def test_mamba_anchor_clamps_to_unit_interval() -> None:
    gate = torch.full((1, 2, 1), 0.9)
    tokens = [["<cot>", "tail"]]
    config = MambaAnchorConfig(enable=True, alpha=0.5)

    adjusted, _ = MambaAnchor(config)(gate, tokens=tokens)
    assert torch.all(adjusted <= 1.0)
    assert torch.all(adjusted >= 0.0)


def test_mamba_anchor_disabled_is_identity() -> None:
    gate = torch.rand(2, 3, 1)
    tokens = torch.tensor([[0, 1, 2], [1, 0, 3]])
    config = MambaAnchorConfig(enable=False)

    adjusted, mask = MambaAnchor(config)(gate, tokens=tokens)
    assert torch.allclose(adjusted, gate)
    assert mask.shape == tokens.shape
