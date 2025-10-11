"""Normalization utilities for latent attention models."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with configurable epsilon."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ResidualNorm(nn.Module):
    """Applies residual connection followed by normalization."""

    def __init__(
        self,
        module: nn.Module,
        norm: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.module = module
        self.norm = norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        residual = x
        out = self.module(x, *args, **kwargs)
        out = self.dropout(out)
        return self.norm(out + residual)


__all__ = ["RMSNorm", "ResidualNorm"]
