"""Dataset loading utilities with synthetic MLA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SineWaveDataset(Dataset):
    """Toy dataset producing sine waves with additive noise."""

    def __init__(self, num_samples: int = 1024, seq_len: int = 128) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.x = torch.linspace(0, 8 * torch.pi, seq_len)

    def __len__(self) -> int:  # type: ignore[override]
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        phase = torch.rand(1) * 2 * torch.pi
        freq = torch.randint(1, 4, (1,), dtype=torch.float32)
        wave = torch.sin(freq * self.x + phase)
        noise = 0.05 * torch.randn_like(wave)
        target = torch.cos(freq * self.x + phase)
        return (wave + noise).unsqueeze(-1), target.unsqueeze(-1)


@dataclass
class LoaderConfig:
    """Configuration for creating dataloaders."""

    batch_size: int = 16
    shuffle: bool = True
    num_workers: int = 0


def create_dataloader(dataset: Dataset, config: LoaderConfig) -> DataLoader:
    """Instantiate a dataloader for the given dataset."""

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )


__all__ = ["SineWaveDataset", "LoaderConfig", "create_dataloader"]
