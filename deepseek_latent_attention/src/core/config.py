"""Configuration objects for hybrid attention modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttnConfig:
    """Configuration for multi-head attention blocks."""

    d_model: int
    n_heads: int
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = True
    causal: bool = False


__all__ = ["AttnConfig"]
