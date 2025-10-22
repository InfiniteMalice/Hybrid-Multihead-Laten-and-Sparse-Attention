"""Minimal training loop scaffolding for latent attention models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn

from ..models.transformer_block import LatentTransformerBlock


@dataclass
class TrainConfig:
    """Configuration controlling the training loop."""

    learning_rate: float = 3e-4
    num_steps: int = 100
    device: str = "cpu"


def make_model(config: Dict) -> nn.Module:
    """Factory function building a :class:`LatentTransformerBlock`."""

    model = LatentTransformerBlock(**config)
    return model


def train(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_cfg: TrainConfig,
    eval_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Run a lightweight training loop for demonstration purposes."""

    device = torch.device(train_cfg.device)
    model.to(device)
    metrics = {}
    for step, (inputs, targets) in enumerate(dataloader):
        if step >= train_cfg.num_steps:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).hidden_states
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if eval_fn is not None:
            metrics = eval_fn(outputs.detach(), targets.detach())
    metrics.setdefault("loss", float(loss.detach()))
    return metrics


def benchmark_against_torch_mha(
    latent_model: nn.Module, torch_mha: nn.Module, inputs: torch.Tensor
) -> Dict[str, float]:
    """Compare FLOPs and runtime with PyTorch's native attention."""

    with torch.autograd.profiler.profile(use_cuda=inputs.is_cuda) as prof_latent:
        latent_model(inputs)
    with torch.autograd.profiler.profile(use_cuda=inputs.is_cuda) as prof_torch:
        torch_mha(inputs, inputs, inputs)
    return {
        "latent_cpu_time": sum(e.cpu_time_total for e in prof_latent.function_events),
        "torch_cpu_time": sum(e.cpu_time_total for e in prof_torch.function_events),
    }


__all__ = ["TrainConfig", "make_model", "train", "benchmark_against_torch_mha"]
