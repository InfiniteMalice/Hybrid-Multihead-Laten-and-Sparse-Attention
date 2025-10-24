#!/usr/bin/env python3
"""Wrapper around EleutherAI's lm-eval-harness with repo defaults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from lm_eval import evaluator

from deepseek_latent_attention.src.utils.config_loader import load_config, resolve_model_tag
from deepseek_latent_attention.src.utils.eval_runtime import (
    prepare_results_dir,
)


def _comma_separated(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_model_args(parameters: Dict[str, str | None]) -> str:
    parts = []
    for key, val in parameters.items():
        if val is None:
            continue
        parts.append(f"{key}={val}")
    return ",".join(parts)


def _build_model_args(runtime: Dict[str, object], overrides: argparse.Namespace) -> str:
    model_name = overrides.model or runtime.get("model_name_or_path")
    tokenizer_name = overrides.tokenizer or runtime.get("tokenizer_name_or_path")
    if model_name is None:
        raise ValueError("Model name must be provided via config or --model")
    dtype = overrides.dtype or runtime.get("dtype", "bfloat16")
    device = overrides.device or runtime.get("device", "cuda")
    revision = overrides.revision or runtime.get("revision")
    parameters = {
        "pretrained": str(model_name),
        "tokenizer": str(tokenizer_name or model_name),
        "dtype": str(dtype),
        "device": str(device),
    }
    if revision:
        parameters["revision"] = str(revision)
    if overrides.use_accelerate:
        parameters["use_accelerate"] = "True"
    if overrides.batch_size:
        parameters["batch_size"] = str(overrides.batch_size)
    return _format_model_args(parameters)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma separated list of lm-eval tasks",
    )
    parser.add_argument("--suite", default="core", help="Results suite name")
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--tokenizer", help="Override tokenizer name")
    parser.add_argument("--dtype", help="Override dtype for model loading")
    parser.add_argument("--device", help="Override target device")
    parser.add_argument("--revision", help="Model revision identifier")
    parser.add_argument("--batch-size", type=int, default=0, help="Per-device batch size override")
    parser.add_argument("--limit", type=int, default=None, help="Optional eval limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for lm-eval")
    parser.add_argument("--fewshot", type=int, default=None, help="Few-shot count")
    parser.add_argument(
        "--use-accelerate",
        action="store_true",
        help="Use accelerate backend for huggingface models",
    )
    parser.add_argument(
        "--model-provider",
        default="hf-causal",
        help="lm-eval model provider identifier",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    runtime = config.get("runtime", {})
    cfg_path = Path(args.cfg)
    model_tag = resolve_model_tag(config, cfg_path.stem)
    suite_dir = prepare_results_dir(args.output_dir, model_tag, args.suite)
    tasks = _comma_separated(args.tasks)
    if not tasks:
        raise ValueError("At least one task must be provided")

    model_args = _build_model_args(runtime, args)

    results = evaluator.simple_evaluate(
        model=args.model_provider,
        model_args=model_args,
        tasks=tasks,
        batch_size=args.batch_size or runtime.get("batch_size", 1),
        limit=args.limit,
        seed=args.seed,
        fewshot=args.fewshot,
    )

    metrics_path = suite_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json_payload = evaluator.serialize_results(results)
        handle.write(json_payload)

    print(f"Saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
