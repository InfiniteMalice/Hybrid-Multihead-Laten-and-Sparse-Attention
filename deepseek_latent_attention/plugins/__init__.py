"""Plugin namespace exporting custom attention heads."""

from __future__ import annotations

from ..src.core.hybrid_mamba_attn import HybridMambaAttention, HybridMambaConfig
from ..src.core.mamba_anchor import MambaAnchor, MambaAnchorConfig
from ..src.core.mamba_block import MambaBlock, MambaConfig
from ..src.core.self_anchor import SelfAnchorConfig, apply_self_anchor, detect_anchors

__all__ = [
    "HybridMambaAttention",
    "HybridMambaConfig",
    "MambaAnchor",
    "MambaAnchorConfig",
    "MambaBlock",
    "MambaConfig",
    "SelfAnchorConfig",
    "apply_self_anchor",
    "detect_anchors",
]
