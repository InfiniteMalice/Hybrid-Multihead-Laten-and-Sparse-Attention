"""Attention gating modules for score scaling prior to softmax."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .gating_config import GatingConfig, GatingMethod


class GatingConfigurationError(ValueError):
    """Raised when gating parameters are misconfigured."""


class GatingRuntimeError(RuntimeError):
    """Raised when gating parameters are missing during execution."""


class GatingModule(nn.Module):
    """Compute gated attention scores.

    Shapes:
        q: ``[B, H, L_q, D]``
        k: ``[B, H, L_k, D]``
        scores: ``[B, H, L_q, L_k]``
    """

    def __init__(self, num_heads: int, head_dim: int, cfg: GatingConfig) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        self.head_gate: Optional[nn.Parameter] = None
        self.token_weight: Optional[nn.Parameter] = None
        self.token_bias: Optional[nn.Parameter] = None

        if cfg.method == GatingMethod.HEADWISE:
            self.head_gate = nn.Parameter(torch.zeros(num_heads))
        elif cfg.method == GatingMethod.TOKENWISE:
            self.token_weight = nn.Parameter(torch.zeros(num_heads, head_dim))
            self.token_bias = nn.Parameter(torch.zeros(num_heads))
        elif cfg.method != GatingMethod.NONE:
            raise GatingConfigurationError(f"Unsupported gating method: {cfg.method}")

    def forward(self, q: Tensor, _k: Tensor, scores: Tensor) -> Tensor:
        """Return gated scores with shape ``[B, H, L_q, L_k]``."""

        if not self.cfg.enabled or self.cfg.method == GatingMethod.NONE:
            return scores

        if self.cfg.method == GatingMethod.HEADWISE:
            if self.head_gate is None:
                raise GatingRuntimeError("Headwise gating parameters are not initialized.")
            gate = torch.sigmoid(self.head_gate + self.cfg.init_bias)
            gate = self._clamp_gate(gate)
            gate = gate.view(1, self.num_heads, 1, 1)
        elif self.cfg.method == GatingMethod.TOKENWISE:
            if self.token_weight is None or self.token_bias is None:
                raise GatingRuntimeError("Tokenwise gating parameters are not initialized.")
            weight = self.token_weight.view(1, self.num_heads, 1, self.head_dim)
            bias = self.token_bias.view(1, self.num_heads, 1)
            logits = (q * weight).sum(dim=-1)
            logits = logits + bias + self.cfg.init_bias
            gate = torch.sigmoid(logits).unsqueeze(-1)
            gate = self._clamp_gate(gate)
        else:
            return scores

        gate = self.dropout(gate)
        return self._apply_gate(scores, gate)

    def _apply_gate(self, scores: Tensor, gate: Tensor) -> Tensor:
        gated = scores * gate
        return torch.where(torch.isfinite(scores), gated, scores)

    def _clamp_gate(self, gate: Tensor) -> Tensor:
        if self.cfg.max_scale <= 0:
            return gate
        return gate.clamp(min=0.0, max=self.cfg.max_scale)


__all__ = ["GatingModule"]
