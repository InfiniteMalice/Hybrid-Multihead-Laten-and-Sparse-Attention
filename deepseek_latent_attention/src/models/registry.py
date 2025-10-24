"""Simple module registry for experiment orchestration."""

"""Registry helpers for model components and feature flags."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

from torch import nn

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}
_FEATURE_FLAGS: Dict[str, Tuple[str, ...]] = {}


def register(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator for registering modules by name."""

    def decorator(cls: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise KeyError(f"Module {name} already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build(name: str, *args, **kwargs) -> nn.Module:
    """Instantiate module by name."""

    if name not in _REGISTRY:
        raise KeyError(f"Module {name} not found in registry: {sorted(_REGISTRY)}")
    return _REGISTRY[name](*args, **kwargs)


def available() -> Dict[str, Callable[..., nn.Module]]:
    """Return a mapping of registered builders."""

    return dict(_REGISTRY)


def register_flags(category: str, values: Iterable[str]) -> None:
    """Register a set of feature flag values for a category."""

    values = tuple(sorted(set(values)))
    _FEATURE_FLAGS[category] = values


def available_flags() -> Dict[str, Tuple[str, ...]]:
    """Return a copy of the registered feature flag definitions."""

    return dict(_FEATURE_FLAGS)


try:  # pragma: no cover - optional plugin import
    from deepseek_latent_attention.plugins import HybridMambaAttention, MambaBlock

    register("hybrid_mamba_attention")(HybridMambaAttention)
    register("mamba_block")(MambaBlock)
except Exception:  # pragma: no cover - registration is optional during tests
    pass


register_flags("BASE", ["MLA_ONLY", "SPARSE_ONLY", "MLA_PLUS_SPARSE"])
register_flags("MAMBA", ["OFF", "MLA", "SPARSE", "BOTH"])
register_flags("ANCHOR_ATTN", ["OFF", "ON"])
register_flags("ANCHOR_MAMBA", ["OFF", "ON"])


__all__ = [
    "register",
    "build",
    "available",
    "register_flags",
    "available_flags",
]
