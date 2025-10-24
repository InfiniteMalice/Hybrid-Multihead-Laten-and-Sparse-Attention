#!/usr/bin/env python3
"""Evaluate ListOps/Text/Retrieval tasks from the Long Range Arena."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from datasets import load_dataset

from deepseek_latent_attention.src.utils.config_loader import load_config, resolve_model_tag
from deepseek_latent_attention.src.utils.eval_runtime import (
    HuggingFaceModelAdapter,
    ModelLoadConfig,
    prepare_results_dir,
    save_json,
)

_TASK_NAMES = ["listops", "text", "retrieval"]


def _load_model(config: Mapping[str, object], overrides: argparse.Namespace) -> HuggingFaceModelAdapter:
    runtime = config.get("runtime", {})
    model_name = overrides.model or runtime.get("model_name_or_path")
    if model_name is None:
        raise ValueError("model_name_or_path must be defined")
    load_cfg = ModelLoadConfig(
        model_name=str(model_name),
        tokenizer_name=str(overrides.tokenizer or runtime.get("tokenizer_name_or_path") or model_name),
        device=str(overrides.device or runtime.get("device", "cuda")),
        dtype=str(overrides.dtype or runtime.get("dtype", "bfloat16")),
        revision=overrides.revision or runtime.get("revision"),
        max_length=int(overrides.max_length or runtime.get("max_length", 4096)),
    )
    return HuggingFaceModelAdapter(load_cfg)


def _extract_choices(dataset) -> List[str]:
    features = getattr(dataset, "features", None)
    if features and "label" in features and hasattr(features["label"], "names"):
        return [str(name) for name in features["label"].names]
    return []


def _format_prompt(sample: Mapping[str, object]) -> str:
    if "input" in sample:
        return str(sample["input"])
    if "text" in sample:
        return str(sample["text"])
    if "sentence" in sample:
        return str(sample["sentence"])
    if "document" in sample and isinstance(sample["document"], Mapping):
        return str(sample["document"].get("text") or sample["document"].get("article") or "")
    if "question" in sample:
        context = sample.get("context") or sample.get("passage") or ""
        return f"{context}\n\nQuestion: {sample['question']}\nAnswer:"
    return str(sample)


def _evaluate_task(
    task: str,
    *,
    model: HuggingFaceModelAdapter,
    split: str,
    limit: Optional[int],
    max_new_tokens: int,
) -> Dict[str, object]:
    try:
        dataset = load_dataset("lra", task, split=split)
    except Exception as exc:  # pragma: no cover - dataset availability
        return {"error": f"{type(exc).__name__}: {exc}"}

    choices = _extract_choices(dataset)
    prompts: List[str] = []
    label_choices: List[Sequence[str]] = []
    targets: List[int] = []

    for sample in dataset:
        prompt = _format_prompt(sample)
        label = sample.get("label")
        if label is None:
            continue
        prompts.append(prompt)
        if choices:
            label_choices.append(choices)
            targets.append(int(label))
        else:
            label_choices.append(["0", "1"])
            targets.append(int(label))
        if limit and len(prompts) >= limit:
            break

    if not prompts:
        return {"error": "No evaluable samples"}

    predictions = model.score_choices(prompts, label_choices)
    accuracy = sum(int(pred == target) for pred, target in zip(predictions, targets)) / len(targets)

    return {
        "metrics": {"accuracy": float(accuracy)},
        "num_samples": len(prompts),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument("--tasks", default=",".join(_TASK_NAMES), help="Comma separated task list")
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--suite", default="lra", help="Results suite name")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task limit")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Tokens per completion")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--tokenizer", help="Override tokenizer name")
    parser.add_argument("--dtype", help="Override dtype")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--revision", help="Model revision identifier")
    parser.add_argument("--max-length", type=int, help="Tokenization maximum length")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    cfg_path = Path(args.cfg)
    model_tag = resolve_model_tag(config, cfg_path.stem)
    suite_dir = prepare_results_dir(args.output_dir, model_tag, args.suite)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("No tasks specified")

    model = _load_model(config, args)

    summary: Dict[str, object] = {"tasks": {}}
    accuracies: List[float] = []

    for task in tasks:
        result = _evaluate_task(
            task,
            model=model,
            split=args.split,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
        summary["tasks"][task] = result
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping) and "accuracy" in metrics:
            accuracies.append(float(metrics["accuracy"]))

    if accuracies:
        summary["macro_accuracy"] = float(statistics.mean(accuracies))

    metrics_path = suite_dir / "metrics.json"
    save_json(metrics_path, summary)
    print(f"Saved LRA results to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
