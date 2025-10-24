#!/usr/bin/env python3
"""Synthetic Needle-in-a-Haystack evaluation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from deepseek_latent_attention.src.utils.config_loader import load_config, resolve_model_tag
from deepseek_latent_attention.src.utils.eval_runtime import (
    HuggingFaceModelAdapter,
    ModelLoadConfig,
    prepare_results_dir,
    save_json,
)


def _load_model(config: Mapping[str, object], overrides: argparse.Namespace) -> HuggingFaceModelAdapter:
    runtime = config.get("runtime", {})
    model_name = overrides.model or runtime.get("model_name_or_path")
    if model_name is None:
        raise ValueError("model_name_or_path must be provided")
    load_cfg = ModelLoadConfig(
        model_name=str(model_name),
        tokenizer_name=str(overrides.tokenizer or runtime.get("tokenizer_name_or_path") or model_name),
        device=str(overrides.device or runtime.get("device", "cuda")),
        dtype=str(overrides.dtype or runtime.get("dtype", "bfloat16")),
        revision=overrides.revision or runtime.get("revision"),
        max_length=int(overrides.max_length or runtime.get("max_length", 4096)),
    )
    return HuggingFaceModelAdapter(load_cfg)


def _make_haystack(depth: int, needle: str, *, rng: random.Random) -> str:
    tokens = [f"token{rng.randint(0, 999)}" for _ in range(depth)]
    index = rng.randrange(depth)
    tokens[index] = needle
    return " ".join(tokens)


def _build_prompts(depth: int, needle: str, runs: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    prompts = []
    template = (
        "Context:\n{context}\n\n"
        "Instruction: identify the hidden phrase and repeat it exactly."
    )
    for _ in range(runs):
        context = _make_haystack(depth, needle, rng=rng)
        prompts.append(template.format(context=context))
    return prompts


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument("--depth", type=int, default=4096, help="Number of tokens in the haystack")
    parser.add_argument("--runs", type=int, default=20, help="Number of synthetic prompts")
    parser.add_argument("--needle", default="NEEDLE_PHRASE", help="Needle string to recover")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Generation budget")
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--suite", default="needle", help="Output suite name")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--tokenizer", help="Override tokenizer name")
    parser.add_argument("--dtype", help="Override dtype")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--revision", help="Model revision identifier")
    parser.add_argument("--max-length", type=int, help="Tokenizer maximum length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    cfg_path = Path(args.cfg)
    model_tag = resolve_model_tag(config, cfg_path.stem)
    suite_dir = prepare_results_dir(args.output_dir, model_tag, args.suite)

    model = _load_model(config, args)

    prompts = _build_prompts(args.depth, args.needle, args.runs, args.seed)
    predictions = model.generate(prompts, max_new_tokens=args.max_new_tokens)

    successes = 0
    for prediction in predictions:
        if args.needle.lower() in prediction.lower():
            successes += 1

    accuracy = successes / len(predictions)
    summary: Dict[str, float | int] = {
        "needle": args.needle,
        "depth": args.depth,
        "runs": args.runs,
        "successes": successes,
        "accuracy": float(accuracy),
    }

    metrics_path = suite_dir / "metrics.json"
    save_json(metrics_path, summary)
    print(f"Saved Needle-in-Haystack metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
