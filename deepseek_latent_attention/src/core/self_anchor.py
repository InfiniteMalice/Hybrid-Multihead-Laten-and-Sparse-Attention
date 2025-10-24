"""Utilities for self-anchored attention biasing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor


TokenLike = Union[
    Tensor,
    Sequence[Union[int, float, str]],
    Sequence[Sequence[Union[int, float, str]]],
    Dict[str, Any],
]


@dataclass
class SelfAnchorConfig:
    """Configuration for self-anchored attention biasing."""

    enable: bool = False
    bias_scale: float = 1.5
    strategy: str = "cot_markers"
    window: int = 64


def detect_anchors(
    tokens: TokenLike,
    strategy: str = "cot_markers",
    window: int = 64,
    *,
    saliency: Optional[Tensor] = None,
) -> Tensor:
    """Return a boolean mask selecting anchor tokens.

    Args:
        tokens: Input tokens expressed as tensors, nested sequences, or dictionaries
            containing ``tokens``/``ids`` and optional ``saliency`` scores.
        strategy: Detection strategy. Supported values are ``cot_markers``, ``regex``,
            and ``saliency_topk``.
        window: Maximum neighbourhood considered when computing anchors. Only
            relevant for ``saliency_topk`` where it caps the number of anchors.
        saliency: Optional precomputed saliency scores ``[B, T]`` used by the
            ``saliency_topk`` strategy.

    Returns:
        A boolean tensor mask of shape ``[B, T]`` where ``True`` marks anchor tokens.
    """

    batch_tokens, numeric_tokens, saliency_scores = _prepare_tokens(tokens, saliency)
    batch_size = len(batch_tokens)
    seq_len = len(batch_tokens[0]) if batch_tokens else 0

    if batch_size == 0:
        return torch.zeros(0, 0, dtype=torch.bool)

    if strategy == "cot_markers":
        mask = _detect_cot_markers(batch_tokens, numeric_tokens)
    elif strategy == "regex":
        mask = _detect_regex(batch_tokens, numeric_tokens)
    elif strategy == "saliency_topk":
        if saliency_scores is None:
            raise ValueError("saliency scores required for saliency_topk strategy")
        limit = min(window, seq_len)
        k = max(1, limit)
        values, indices = torch.topk(saliency_scores, k=k, dim=-1)
        mask = torch.zeros_like(saliency_scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        # Guard against duplicates when k == seq_len by ensuring only the top window
        # positions remain marked.
        if limit < seq_len:
            threshold = values[..., -1:].expand_as(saliency_scores)
            mask &= saliency_scores >= threshold
    else:
        raise ValueError(f"Unsupported anchor strategy: {strategy}")

    return mask


def apply_self_anchor(
    scores: Tensor,
    tokens: TokenLike,
    config: SelfAnchorConfig,
    mask: Optional[Tensor] = None,
    *,
    saliency: Optional[Tensor] = None,
    return_stats: bool = False,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    """Apply self-anchoring bias to attention scores."""

    if not config.enable:
        return scores, {"anchor_mask": None} if return_stats else None

    anchor_mask = detect_anchors(
        tokens, strategy=config.strategy, window=config.window, saliency=saliency
    )
    anchor_bias = anchor_mask.to(device=scores.device, dtype=scores.dtype)
    anchor_bias = anchor_bias.unsqueeze(1).unsqueeze(1) * config.bias_scale

    if mask is not None:
        broadcast_mask = mask
        if broadcast_mask.dtype == torch.bool:
            broadcast_mask = broadcast_mask.to(dtype=scores.dtype)
        anchor_bias = anchor_bias * broadcast_mask

    biased_scores = scores + anchor_bias

    stats: Optional[Dict[str, Tensor]] = None
    if return_stats:
        stats = {"anchor_mask": anchor_mask}

    return biased_scores, stats


def _prepare_tokens(
    tokens: TokenLike, saliency: Optional[Tensor]
) -> Tuple[List[List[Any]], Optional[Tensor], Optional[Tensor]]:
    if isinstance(tokens, dict):
        payload = tokens.get("tokens") or tokens.get("ids")
        if payload is None:
            raise ValueError("token dictionary must contain 'tokens' or 'ids'")
        saliency_scores = tokens.get("saliency")
        return _prepare_tokens(payload, saliency_scores if saliency is None else saliency)

    if isinstance(tokens, Tensor):
        tensor_tokens = tokens
        if tensor_tokens.dim() == 1:
            tensor_tokens = tensor_tokens.unsqueeze(0)
        batch_tokens = [list(map(_coerce_scalar, row)) for row in tensor_tokens.tolist()]
        numeric_tokens = tensor_tokens.to(dtype=torch.long)
        saliency_scores = saliency
        if saliency_scores is not None and saliency_scores.dim() == 1:
            saliency_scores = saliency_scores.unsqueeze(0)
        if saliency_scores is not None and saliency_scores.shape != numeric_tokens.shape:
            raise ValueError("saliency scores must match token shape")
        return batch_tokens, numeric_tokens, saliency_scores

    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
        if len(tokens) == 0:
            raise ValueError("tokens cannot be empty")
        first = tokens[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            batch_tokens = [list(map(_coerce_scalar, seq)) for seq in tokens]
        else:
            batch_tokens = [list(map(_coerce_scalar, tokens))]

        numeric_tokens: Optional[Tensor]
        if isinstance(batch_tokens[0][0], (int, float)):
            numeric_tokens = torch.tensor(batch_tokens, dtype=torch.long)
        else:
            numeric_tokens = None

        saliency_scores = saliency
        if saliency_scores is not None:
            saliency_scores = _coerce_saliency(saliency_scores, len(batch_tokens))
            if numeric_tokens is not None and saliency_scores.shape != numeric_tokens.shape:
                raise ValueError("saliency scores must align with numeric tokens")

        return batch_tokens, numeric_tokens, saliency_scores

    raise TypeError(f"Unsupported token container: {type(tokens)!r}")


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.item()
    return value


def _coerce_saliency(saliency: Any, batch_size: int) -> Tensor:
    if isinstance(saliency, Tensor):
        if saliency.dim() == 1:
            saliency = saliency.unsqueeze(0)
        return saliency.to(dtype=torch.float32)

    if isinstance(saliency, Sequence):
        if len(saliency) == 0:
            raise ValueError("saliency cannot be empty")
        if isinstance(saliency[0], Sequence):
            return torch.tensor(saliency, dtype=torch.float32)
        return torch.tensor([saliency] * batch_size, dtype=torch.float32)

    raise TypeError("Unsupported saliency container")


def _detect_cot_markers(
    batch_tokens: List[List[Any]], numeric_tokens: Optional[Tensor]
) -> Tensor:
    batch_size = len(batch_tokens)
    seq_len = len(batch_tokens[0])
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for b_idx, sequence in enumerate(batch_tokens):
        for t_idx, token in enumerate(sequence):
            if isinstance(token, str):
                lower = token.lower()
                if any(marker in lower for marker in ("cot", "answer", "anchor", "<", "[") ):
                    mask[b_idx, t_idx] = True
            else:
                value = int(token)
                if value <= 3 or value < 0:
                    mask[b_idx, t_idx] = True

    if numeric_tokens is not None:
        numeric_mask = torch.isin(numeric_tokens, torch.tensor([0, 1, 2, 3, 50256]))
        mask |= numeric_mask

    return mask


def _detect_regex(
    batch_tokens: List[List[Any]], numeric_tokens: Optional[Tensor]
) -> Tensor:
    batch_size = len(batch_tokens)
    seq_len = len(batch_tokens[0])
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    pattern = re.compile(r"([\d]+[:\.]$|[:;,.!?]|^#|^\[|^<)")

    for b_idx, sequence in enumerate(batch_tokens):
        for t_idx, token in enumerate(sequence):
            if isinstance(token, str):
                if pattern.search(token):
                    mask[b_idx, t_idx] = True
            else:
                value = abs(int(token))
                if value % 10 == 0:
                    mask[b_idx, t_idx] = True

    if numeric_tokens is not None:
        modulo_mask = (numeric_tokens.abs() % 10) == 0
        mask |= modulo_mask

    return mask


__all__ = ["SelfAnchorConfig", "detect_anchors", "apply_self_anchor"]
