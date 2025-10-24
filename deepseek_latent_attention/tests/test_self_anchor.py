"""Tests for self-anchor attention utilities."""

from __future__ import annotations

import torch

from deepseek_latent_attention.src.core.self_anchor import (
    SelfAnchorConfig,
    apply_self_anchor,
    detect_anchors,
)


def test_self_anchor_bias_increases_anchor_mass() -> None:
    scores = torch.zeros(1, 1, 4, 4)
    tokens = [["<cot>", "token", "answer", "rest"]]
    config = SelfAnchorConfig(enable=True, bias_scale=2.0)

    baseline = torch.softmax(scores, dim=-1)
    biased_scores, stats = apply_self_anchor(scores, tokens, config, return_stats=True)
    assert stats is not None and stats["anchor_mask"] is not None
    anchor_mask = stats["anchor_mask"].to(dtype=baseline.dtype)

    updated = torch.softmax(biased_scores, dim=-1)
    anchor_mass_before = (baseline * anchor_mask.unsqueeze(1).unsqueeze(1)).sum()
    anchor_mass_after = (updated * anchor_mask.unsqueeze(1).unsqueeze(1)).sum()

    assert anchor_mass_after > anchor_mass_before


def test_self_anchor_disabled_is_noop() -> None:
    scores = torch.randn(2, 2, 3, 3)
    tokens = torch.tensor([[0, 1, 2], [2, 3, 4]])
    config = SelfAnchorConfig(enable=False)

    biased_scores, stats = apply_self_anchor(scores, tokens, config, return_stats=True)
    assert torch.allclose(biased_scores, scores)
    assert stats is not None and stats["anchor_mask"] is None


def test_detect_anchors_saliency_topk() -> None:
    tokens = torch.tensor([[5, 7, 9, 11]])
    saliency = torch.tensor([[0.1, 0.9, 0.2, 0.8]])

    mask = detect_anchors(tokens, strategy="saliency_topk", window=2, saliency=saliency)

    expected = torch.tensor([[False, True, False, True]])
    assert torch.equal(mask, expected)
