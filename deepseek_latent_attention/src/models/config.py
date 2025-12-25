"""Configuration objects for latent attention experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.gating_config import GatingConfig


@dataclass
class LatentAttentionConfig:
    """Minimal configuration for training latent transformer blocks."""

    embed_dim: int = 512
    num_heads: int = 8
    latent_dim_ratio: float = 0.25
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_sparse: bool = False
    use_bias: bool = True
    latent_sparse_topk: int | None = None
    gating: GatingConfig = field(default_factory=GatingConfig)


__all__ = ["LatentAttentionConfig"]
