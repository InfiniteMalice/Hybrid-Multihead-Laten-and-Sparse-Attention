"""Fusion layers combining latent and sparse attention pathways."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .mha_latent import LatentAttention, LatentSparseAttention, AttentionStats


class LatentSparseFusion(nn.Module):
    """Fuse dense latent attention with its sparse counterpart.

    This layer is useful for ablation studies where one may wish to interpolate
    between dense MLA and a sparse-regularized variant. The outputs from both
    branches are combined via a learnable scalar gate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim_ratio: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent = LatentAttention(embed_dim, num_heads, latent_dim_ratio, dropout)
        self.sparse = LatentSparseAttention(embed_dim, num_heads, latent_dim_ratio, dropout)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sparse_config: Optional[Dict[str, torch.Tensor]] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], AttentionStats]:
        dense_out, dense_weights, dense_stats = self.latent(
            query, key, value, attention_mask=attention_mask, need_weights=True
        )
        sparse_out, sparse_weights, sparse_stats = self.sparse(
            query,
            key,
            value,
            attention_mask=attention_mask,
            sparse_config=sparse_config,
            need_weights=True,
        )
        gate = torch.sigmoid(self.gate)
        output = gate * dense_out + (1 - gate) * sparse_out
        if need_weights:
            weights = gate * dense_weights + (1 - gate) * sparse_weights
        else:
            weights = None
        fused_stats = AttentionStats()
        if dense_stats.head_entropy is not None and sparse_stats.head_entropy is not None:
            fused_stats.head_entropy = (
                gate * dense_stats.head_entropy + (1 - gate) * sparse_stats.head_entropy
            )
        if dense_stats.sparsity is not None and sparse_stats.sparsity is not None:
            fused_stats.sparsity = gate * dense_stats.sparsity + (1 - gate) * sparse_stats.sparsity
        return output, weights, fused_stats


__all__ = ["LatentSparseFusion"]
