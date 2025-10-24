"""Minimal selective-scan style state space module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class MambaConfig:
    """Configuration for the minimal Mamba block."""

    d_model: int
    state_size: int = 64
    dt_rank: int = 16
    conv_kernel_size: int = 4
    dropout: float = 0.0


class MambaBlock(nn.Module):
    """Selective scan inspired state space module.

    The implementation purposefully remains lightweight: it applies a depthwise
    temporal convolution to the input, projects it into a latent state, and scans
    the sequence recurrently using a single linear transition. This is sufficient
    for experimentation and unit testing while keeping the codebase dependency
    free.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        if config.conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be >= 1")

        self.config = config
        self.input_proj = nn.Linear(config.d_model, config.state_size)
        self.delta_proj = nn.Linear(config.d_model, config.dt_rank)
        self.conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,
            groups=config.d_model,
            bias=False,
        )
        self.transition = nn.Linear(config.dt_rank, config.state_size, bias=False)
        self.output_proj = nn.Linear(config.state_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.SiLU()

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.xavier_uniform_(self.transition.weight)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Run the state space block.

        Args:
            x: Input tensor of shape ``[B, T, D]``.

        Returns:
            Output tensor of identical shape ``[B, T, D]``.
        """

        batch, seq_len, d_model = x.shape
        if d_model != self.config.d_model:
            raise ValueError("Input feature dimension mismatch")

        conv_in = x.transpose(1, 2)
        conv_out = self.conv(conv_in)
        conv_out = conv_out[..., :seq_len]
        conv_out = conv_out.transpose(1, 2)

        projected = self.input_proj(x + conv_out)
        delta = torch.tanh(self.delta_proj(x))
        delta = self.transition(delta)

        state = torch.zeros(batch, self.config.state_size, device=x.device, dtype=x.dtype)
        outputs = []
        for timestep in range(seq_len):
            update = projected[:, timestep]
            state = self.activation(delta[:, timestep] + state)
            state = state + update
            outputs.append(self.output_proj(self.dropout(state)))

        stacked = torch.stack(outputs, dim=1)
        return stacked


__all__ = ["MambaConfig", "MambaBlock"]
