"""Evaluation metrics for latent attention experiments."""
from __future__ import annotations

from typing import Dict

import torch


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute simple regression metrics."""

    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    return {"mse": mse, "mae": mae}


__all__ = ["compute_metrics"]
