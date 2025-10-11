"""Transformer block using latent attention modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..core.mha_latent import LatentAttention, LatentSparseAttention
from ..core.normalization import RMSNorm


@dataclass
class BlockOutputs:
    """Captures outputs and attention diagnostics for a forward pass."""

    hidden_states: torch.Tensor
    attn_weights: Optional[torch.Tensor]
    attn_stats: Optional[dict]


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class LatentTransformerBlock(nn.Module):
    """Single transformer block with optional sparse latent attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        latent_dim_ratio: float = 0.25,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_sparse: bool = False,
    ) -> None:
        super().__init__()
        attn_cls = LatentSparseAttention if use_sparse else LatentAttention
        self.attn = attn_cls(embed_dim, num_heads, latent_dim_ratio, dropout, track_stats=True)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        act_cls = _ACTIVATIONS.get(activation.lower(), nn.GELU)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.mlp_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sparse_config: Optional[dict] = None,
    ) -> BlockOutputs:
        attn_out, attn_weights, stats = self.attn(
            x,
            x,
            x,
            attention_mask=attention_mask,
            sparse_config=sparse_config,
            need_weights=True,
        )
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp_drop(self.mlp(x))
        x = self.norm2(x + mlp_out)
        stat_dict = {
            "entropy": stats.head_entropy.detach().cpu() if stats.head_entropy is not None else None,
            "sparsity": stats.sparsity.detach().cpu() if stats.sparsity is not None else None,
        }
        return BlockOutputs(hidden_states=x, attn_weights=attn_weights, attn_stats=stat_dict)


__all__ = ["LatentTransformerBlock", "BlockOutputs"]
