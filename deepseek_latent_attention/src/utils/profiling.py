"""Profiling helpers for measuring FLOPs and sparsity."""

from __future__ import annotations

from typing import Dict

import torch


def count_flops(attn_weights: torch.Tensor, latent_dim: int) -> int:
    """Estimate FLOPs for latent attention operation."""

    b, h, q, k = attn_weights.shape
    return int(2 * b * h * q * latent_dim)


def summarize_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Convert attention statistics tensors into scalars for logging."""

    return {k: float(v.mean()) for k, v in stats.items() if v is not None}


__all__ = ["count_flops", "summarize_stats"]
