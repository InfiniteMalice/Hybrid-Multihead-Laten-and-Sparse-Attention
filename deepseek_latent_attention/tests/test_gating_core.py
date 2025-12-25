"""Unit tests for attention gating modules."""

from __future__ import annotations

import pytest

try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled during CI
    pytest.skip("PyTorch is required for gating tests", allow_module_level=True)

from deepseek_latent_attention.src.core.gating import GatingModule
from deepseek_latent_attention.src.core.gating_config import GatingConfig, GatingMethod


@pytest.mark.parametrize("method", [GatingMethod.HEADWISE, GatingMethod.TOKENWISE])
def test_gating_preserves_shape(method: GatingMethod) -> None:
    cfg = GatingConfig(method=method, enabled=True)
    module = GatingModule(num_heads=2, head_dim=4, cfg=cfg)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 5, 4)
    scores = torch.randn(1, 2, 3, 5)

    out = module(q, k, scores)

    assert out.shape == scores.shape


def test_gating_no_op_when_disabled() -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    scores = torch.randn(1, 2, 3, 3)

    cfg_disabled = GatingConfig(method=GatingMethod.HEADWISE, enabled=False)
    disabled_module = GatingModule(num_heads=2, head_dim=4, cfg=cfg_disabled)
    out_disabled = disabled_module(q, k, scores)

    cfg_none = GatingConfig(method=GatingMethod.NONE, enabled=True)
    none_module = GatingModule(num_heads=2, head_dim=4, cfg=cfg_none)
    out_none = none_module(q, k, scores)

    assert torch.allclose(out_disabled, scores)
    assert torch.allclose(out_none, scores)


def test_headwise_gating_scales_each_head() -> None:
    cfg = GatingConfig(method=GatingMethod.HEADWISE, enabled=True, max_scale=2.0)
    module = GatingModule(num_heads=2, head_dim=4, cfg=cfg)
    target = torch.tensor([0.5, 0.8])
    module.head_gate.data.copy_(torch.logit(target))

    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 2, 4)
    scores = torch.ones(1, 2, 1, 2)

    out = module(q, k, scores)
    expected = scores * target.view(1, 2, 1, 1)
    assert torch.allclose(out, expected)


def test_tokenwise_gating_scales_query_positions() -> None:
    cfg = GatingConfig(method=GatingMethod.TOKENWISE, enabled=True, max_scale=2.0)
    module = GatingModule(num_heads=1, head_dim=2, cfg=cfg)
    module.token_weight.data.copy_(torch.tensor([[1.0, 0.0]]))
    module.token_bias.data.zero_()

    q = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
    k = torch.zeros(1, 1, 2, 2)
    scores = torch.ones(1, 1, 2, 2)

    out = module(q, k, scores)
    gates = torch.sigmoid(torch.tensor([[0.0, 1.0]])).view(1, 1, 2, 1)
    expected = scores * gates
    assert torch.allclose(out, expected)


def test_gating_gradients_flow() -> None:
    cfg = GatingConfig(method=GatingMethod.TOKENWISE, enabled=True)
    module = GatingModule(num_heads=1, head_dim=3, cfg=cfg)
    q = torch.randn(2, 1, 4, 3, requires_grad=True)
    k = torch.randn(2, 1, 4, 3)
    scores = torch.randn(2, 1, 4, 4, requires_grad=True)

    out = module(q, k, scores)
    loss = out.sum()
    loss.backward()

    assert module.token_weight.grad is not None
    assert module.token_bias.grad is not None
