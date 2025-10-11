"""Top-level package for DeepSeek-style latent attention modules."""

from .core.mha_latent import LatentAttention, LatentSparseAttention

__all__ = [
    "LatentAttention",
    "LatentSparseAttention",
]
