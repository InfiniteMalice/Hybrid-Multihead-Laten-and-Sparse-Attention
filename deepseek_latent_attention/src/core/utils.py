"""Utility helpers for attention modules."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def create_mask(
    seq_len: int,
    causal: bool = False,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a broadcastable attention mask.

    Args:
        seq_len: Sequence length ``T``.
        causal: When ``True`` returns a causal mask.
        device: Optional device for the mask tensor.

    Returns:
        Boolean mask with shape ``[1, 1, T, T]``.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    if causal:
        mask = torch.tril(mask)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


__all__ = ["create_mask"]
