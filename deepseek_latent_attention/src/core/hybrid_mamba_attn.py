"""Fusion utilities for attention and Mamba style state space modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .mamba_block import MambaBlock
from .mamba_anchor import MambaAnchor, MambaAnchorConfig
from .self_anchor import SelfAnchorConfig, apply_self_anchor, detect_anchors


@dataclass
class HybridMambaConfig:
    """Configuration toggles for the hybrid Mamba attention head."""

    enable: bool = False
    apply_to: str = "mla"  # choices: mla | sparse | both
    gate_bias: float = 0.0


class HybridMambaAttention(nn.Module):
    """Fuse attention output with a Mamba state space branch."""

    def __init__(
        self,
        base_attn: nn.Module,
        mamba: Optional[MambaBlock],
        *,
        config: HybridMambaConfig,
        self_anchor: Optional[SelfAnchorConfig] = None,
        mamba_anchor: Optional[MambaAnchorConfig] = None,
    ) -> None:
        super().__init__()
        self.base_attn = base_attn
        self.mamba = mamba
        self.config = config
        self.self_anchor_config = self_anchor
        self.mamba_anchor = MambaAnchor(mamba_anchor) if mamba_anchor else None

        d_model = getattr(base_attn, "d_model", None)
        if d_model is None and hasattr(base_attn, "config"):
            d_model = getattr(base_attn.config, "d_model", None)
        if d_model is None:
            raise ValueError("base_attn must expose a d_model attribute")

        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, config.gate_bias)

    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        tokens: Optional[Tensor] = None,
        saliency: Optional[Tensor] = None,
        return_stats: bool = False,
        **attn_kwargs,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if not self.config.enable:
            output, weights = self.base_attn(
                x, mask=mask, key_padding_mask=key_padding_mask, **attn_kwargs
            )
            stats = {}
            if return_stats:
                stats = {
                    "gate": torch.zeros_like(x),
                    "ssm_norm": torch.zeros(x.size(0), x.size(1), 1, device=x.device),
                    "attn_entropy": torch.zeros(x.size(0), x.size(1), device=x.device),
                    "meta_token_indices": torch.empty(0, 2, dtype=torch.long, device=x.device),
                }
            return output, stats

        base_out, attn_weights = self.base_attn(
            x,
            mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_stats,
            **attn_kwargs,
        )

        if self.self_anchor_config and self.self_anchor_config.enable and attn_weights is not None:
            scores = attn_weights.unsqueeze(1)
            biased_scores, stats = apply_self_anchor(
                scores,
                tokens if tokens is not None else x.argmax(dim=-1),
                self.self_anchor_config,
                mask=None,
                saliency=saliency,
                return_stats=True,
            )
            attn_weights = biased_scores.squeeze(1)
            anchor_mask = stats["anchor_mask"]
        else:
            anchor_mask = (
                detect_anchors(
                    tokens if tokens is not None else x.argmax(dim=-1),
                    strategy=self.self_anchor_config.strategy if self.self_anchor_config else "cot_markers",
                )
                if tokens is not None or self.self_anchor_config
                else None
            )

        if self.mamba is None:
            raise ValueError("Hybrid Mamba requires an instantiated MambaBlock")
        ssm_out = self.mamba(x)
        gate = torch.sigmoid(self.gate_proj(x))

        if self.mamba_anchor is not None:
            gate, anchor_mask = self.mamba_anchor(
                gate,
                tokens=tokens if tokens is not None else x.argmax(dim=-1),
                anchor_mask=anchor_mask,
            )

        fused = gate * ssm_out + (1.0 - gate) * base_out

        stats: Dict[str, Tensor] = {}
        if return_stats:
            stats["gate"] = gate.detach()
            stats["ssm_norm"] = ssm_out.norm(dim=-1, keepdim=True)
            if attn_weights is not None:
                probs = attn_weights.clamp_min(1e-6)
                entropy = -(probs * probs.log()).sum(dim=-1)
                stats["attn_entropy"] = entropy
            else:
                stats["attn_entropy"] = torch.zeros_like(x[..., 0])
            if anchor_mask is not None:
                indices = anchor_mask.nonzero(as_tuple=False)
                stats["meta_token_indices"] = indices.to(device=x.device)
            else:
                stats["meta_token_indices"] = torch.empty(0, 2, dtype=torch.long, device=x.device)

        return fused, stats


__all__ = ["HybridMambaConfig", "HybridMambaAttention"]
