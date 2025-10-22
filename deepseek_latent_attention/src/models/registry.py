"""Simple module registry for experiment orchestration."""

from __future__ import annotations

from typing import Callable, Dict, Type

from torch import nn

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


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


__all__ = ["register", "build", "available"]
