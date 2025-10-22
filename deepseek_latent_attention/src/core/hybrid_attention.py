"""Hybrid multi-head attention building blocks."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:  # pragma: no cover - optional dependency guard
    from einops import rearrange
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without einops

    def rearrange(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
        """Minimal fallback for ``einops.rearrange`` used in this module."""

        expected = "b t (three h d) -> three b h t d"
        if pattern != expected:
            raise ImportError(f"einops is required for pattern: {pattern!r}")

        three = axes_lengths.get("three")
        heads = axes_lengths.get("h")
        dim = axes_lengths.get("d")
        if three is None or heads is None or dim is None:
            raise ImportError("einops is required when axes lengths are unspecified")

        batch, seq_len, _ = tensor.shape
        tensor = tensor.view(batch, seq_len, three, heads, dim)
        tensor = tensor.permute(2, 0, 3, 1, 4)
        return tensor


from .config import AttnConfig
from .utils import create_mask


class MultiheadAttn(nn.Module):
    """Multi-head attention with configurable masking.

    Args:
        config: Attention configuration dataclass.
    """

    def __init__(self, config: AttnConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.attn_dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the multi-head attention block.

        Args:
            x: Input tensor ``[B, T, D]``.
            mask: Optional attn mask ``[B|1, H|1, T, T]`` or broadcastable boolean tensor.
            key_padding_mask: Optional padding mask ``[B, T]`` where ``True`` marks pads.
            need_weights: Return averaged attention weights when ``True``.

        Returns:
            Tuple containing the output ``[B, T, D]`` and optional weights ``[B, T, T]``.
        """
        batch, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads, d=self.d_k
        )
        q, k, v = qkv.unbind(dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        mask = self._resolve_mask(mask, batch, seq_len, x.device)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        if key_padding_mask is not None:
            scores = self._apply_padding_mask(scores, key_padding_mask)

        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        attn_out = torch.matmul(probs, v)

        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(batch, seq_len, self.d_model)
        output = self.out(attn_out)
        output = self.out_dropout(output)

        weights = probs.mean(dim=1) if need_weights else None
        return output, weights

    def _resolve_mask(
        self,
        mask: Optional[Tensor],
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if mask is None and self.config.causal:
            mask = create_mask(seq_len, causal=True, device=device)

        if mask is None:
            return None

        mask = mask.to(device=device)
        if mask.dtype != torch.bool:
            mask = mask != 0

        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError("mask must have 2, 3, or 4 dimensions")

        if mask.shape[-2:] != (seq_len, seq_len):
            raise ValueError("mask must match sequence length")

        if mask.shape[0] not in (1, batch):
            raise ValueError("mask batch dimension mismatch")

        if mask.shape[1] not in (1, self.n_heads):
            raise ValueError("mask head dimension mismatch")

        mask = mask.expand(batch, self.n_heads, seq_len, seq_len)
        return mask

    def _apply_padding_mask(self, scores: Tensor, padding_mask: Tensor) -> Tensor:
        padding_mask = padding_mask.to(device=scores.device)
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask != 0

        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(-1, self.n_heads, scores.size(-2), -1)
        scores = scores.masked_fill(padding_mask, float("-inf"))
        return scores


__all__ = ["MultiheadAttn"]
