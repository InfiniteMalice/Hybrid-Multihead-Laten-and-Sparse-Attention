"""Configuration objects for attention gating."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GatingMethod(str, Enum):
    """Supported gating modes for attention score scaling."""

    HEADWISE = "headwise"
    TOKENWISE = "tokenwise"
    NONE = "none"


@dataclass
class GatingConfig:
    """Configuration for optional attention score gating."""

    method: GatingMethod = GatingMethod.NONE
    enabled: bool = False
    init_bias: float = 0.0
    max_scale: float = 1.5
    dropout: float = 0.0


__all__ = ["GatingConfig", "GatingMethod"]
