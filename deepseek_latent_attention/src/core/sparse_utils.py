"""Utilities for sparse masking strategies compatible with latent attention."""
from __future__ import annotations

from typing import Dict, Optional

import torch

SparseConfig = Dict[str, torch.Tensor]


def _validate_scores(scores: torch.Tensor) -> None:
    if scores.dim() != 4:
        raise ValueError("Scores must be of shape (B, H, T, T)")


def block_sparse_mask(
    seq_len: int,
    block_size: int,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    """Generate a block-sparse mask allowing local attention windows.

    Args:
        seq_len: Sequence length.
        block_size: Number of tokens per block (both query and key side).
        device: Optional device the mask should live on.
    Returns:
        Boolean tensor of shape ``(seq_len, seq_len)`` where ``False`` entries
        indicate masked pairs.
    """

    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    idx = torch.arange(seq_len, device=device)
    block_id = idx // block_size
    mask = block_id.unsqueeze(-1) == block_id.unsqueeze(0)
    return mask


def topk_sparse_mask(scores: torch.Tensor, k: int) -> SparseConfig:
    """Construct a sparse configuration keeping top-k keys per query.

    Args:
        scores: Attention scores tensor ``(B, H, T, T)``.
        k: Number of keys to keep for each query position.
    Returns:
        Dict containing indices and values describing the sparse pattern.
    """

    _validate_scores(scores)
    if k <= 0:
        raise ValueError("k must be positive")
    top_values, top_indices = torch.topk(scores, k=k, dim=-1)
    mask = torch.full_like(scores, float("-inf"))
    mask.scatter_(-1, top_indices, top_values)
    return {"masked_scores": mask}


def apply_sparse_mask(scores: torch.Tensor, sparse_cfg: Optional[SparseConfig]) -> torch.Tensor:
    """Apply sparse configuration to attention scores prior to softmax."""

    if sparse_cfg is None:
        return scores
    if "masked_scores" in sparse_cfg:
        masked = sparse_cfg["masked_scores"]
        if masked.shape != scores.shape:
            raise ValueError("Sparse mask shape mismatch")
        return masked
    if "binary_mask" in sparse_cfg:
        mask = sparse_cfg["binary_mask"].to(dtype=scores.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(scores)
        return scores.masked_fill(~mask.bool(), float("-inf"))
    raise KeyError("Unsupported sparse configuration provided")


def fuse_block_topk(
    block_mask: torch.Tensor,
    topk_cfg: SparseConfig,
) -> SparseConfig:
    """Combine block-sparse and top-k sparse rules into one configuration."""

    binary_mask = block_mask.clone()
    if "binary_mask" in topk_cfg:
        binary_mask &= topk_cfg["binary_mask"].bool()
        return {"binary_mask": binary_mask}
    masked_scores = topk_cfg.get("masked_scores")
    if masked_scores is None:
        raise KeyError("topk_cfg must provide either binary_mask or masked_scores")
    mask = binary_mask.unsqueeze(0).unsqueeze(0).to(dtype=masked_scores.dtype)
    return {"masked_scores": masked_scores.masked_fill(~mask.bool(), float("-inf"))}


__all__ = [
    "SparseConfig",
    "block_sparse_mask",
    "topk_sparse_mask",
    "apply_sparse_mask",
    "fuse_block_topk",
]
