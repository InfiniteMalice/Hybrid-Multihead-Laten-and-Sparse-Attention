"""Gate adjustment helpers for Mamba fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .self_anchor import TokenLike, detect_anchors


@dataclass
class MambaAnchorConfig:
    """Configuration for nudging Mamba gates towards anchors."""

    enable: bool = False
    alpha: float = 0.1
    strategy: str = "cot_markers"
    window: int = 64


class MambaAnchor:
    """Utility module applying anchor-based gate nudging."""

    def __init__(self, config: MambaAnchorConfig) -> None:
        self.config = config

    def __call__(
        self,
        gate: Tensor,
        *,
        tokens: TokenLike,
        anchor_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if not self.config.enable:
            if anchor_mask is None:
                anchor_mask = torch.zeros(
                    gate.size(0), gate.size(1), dtype=torch.bool, device=gate.device
                )
            return gate, anchor_mask

        if anchor_mask is None:
            anchor_mask = detect_anchors(
                tokens, strategy=self.config.strategy, window=self.config.window
            )
        anchor_bias = anchor_mask.to(device=gate.device, dtype=gate.dtype)
        anchor_bias = anchor_bias.unsqueeze(-1) * self.config.alpha
        adjusted = torch.clamp(gate + anchor_bias, 0.0, 1.0)
        return adjusted, anchor_mask


__all__ = ["MambaAnchor", "MambaAnchorConfig"]
