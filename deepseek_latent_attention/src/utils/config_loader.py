"""Utility helpers for experiment configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def _merge_dicts(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``update`` into ``base``."""

    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            base[key] = _merge_dicts(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = path.read_text()
    data = yaml.safe_load(payload) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration at {path} must be a mapping")
    return data


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a configuration file resolving ``extends`` chains."""

    cfg_path = Path(path).expanduser().resolve()
    data = _load_yaml(cfg_path)
    parents: Iterable[str] = data.pop("extends", []) or []
    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = (cfg_path.parent / parent).resolve()
        parent_data = load_config(parent_path)
        merged = _merge_dicts(merged, parent_data)
    merged = _merge_dicts(merged, data)
    return merged


def resolve_model_tag(config: Mapping[str, Any], fallback: str) -> str:
    """Determine the model tag used for filesystem outputs."""

    name = config.get("name")
    if isinstance(name, str) and name:
        return name
    runtime = config.get("runtime", {})
    if isinstance(runtime, Mapping):
        tag = runtime.get("model_tag")
        if isinstance(tag, str) and tag:
            return tag
    return fallback


__all__ = ["load_config", "resolve_model_tag"]
