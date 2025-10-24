#!/usr/bin/env python3
"""Evaluate models on LongBench tasks with consistent logging."""

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

_ROUGE_TASKS = {
    "gov_report",
    "multi_news",
    "qmsum",
    "vcsum",
}

_DEFAULT_TASKS = [
    "qasper",
    "hotpotqa",
    "multifieldqa_en",
    "narrativeqa",
    "gov_report",
    "qmsum",
]


def _tokenize(text: str) -> List[str]:
    tokens = [token.strip(".,;!?\"'()[]{}") for token in text.lower().split()]
    return [tok for tok in tokens if tok]


def _f1_score(prediction: str, references: Sequence[str]) -> float:
    pred_tokens = _tokenize(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue
        overlap = len(set(pred_tokens) & set(ref_tokens))
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def _summarize(prompts: Sequence[str], model: HuggingFaceModelAdapter, max_tokens: int) -> List[str]:
    outputs = model.generate(prompts, max_new_tokens=max_tokens)
    trimmed: List[str] = []
    for output, prompt in zip(outputs, prompts):
        if output.startswith(prompt):
            trimmed.append(output[len(prompt) :].strip())
        else:
            trimmed.append(output.strip())
    return trimmed


def _build_prompt(sample: Mapping[str, object]) -> str:
    context = str(sample.get("input") or sample.get("context") or sample.get("document") or sample.get("article") or "")
    question = sample.get("question") or sample.get("query")
    if isinstance(question, Mapping):
        question = question.get("text")
    if question:
        return f"{context}\n\nQuestion: {question}\nAnswer:"
    return context


def _extract_references(sample: Mapping[str, object]) -> List[str]:
    if "answer" in sample:
        ans = sample["answer"]
    elif "answers" in sample:
        ans = sample["answers"]
    elif "outputs" in sample:
        ans = sample["outputs"]
    elif "target" in sample:
        ans = sample["target"]
    else:
        return []
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, Mapping):
        values = ans.get("text") or ans.get("answers")
        if isinstance(values, str):
            return [values]
        if isinstance(values, Sequence):
            return [str(v) for v in values]
        return []
    if isinstance(ans, Sequence):
        return [str(item) for item in ans]
    return []


def _load_model(config: Mapping[str, object], overrides: argparse.Namespace) -> HuggingFaceModelAdapter:
    runtime = config.get("runtime", {})
    model_name = overrides.model or runtime.get("model_name_or_path")
    if model_name is None:
        raise ValueError("model_name_or_path must be defined in the config or via --model")
    load_cfg = ModelLoadConfig(
        model_name=str(model_name),
        tokenizer_name=str(overrides.tokenizer or runtime.get("tokenizer_name_or_path") or model_name),
        device=str(overrides.device or runtime.get("device", "cuda")),
        dtype=str(overrides.dtype or runtime.get("dtype", "bfloat16")),
        revision=overrides.revision or runtime.get("revision"),
        max_length=int(overrides.max_length or runtime.get("max_length", 4096)),
    )
    return HuggingFaceModelAdapter(load_cfg)


def _evaluate_task(
    dataset_name: str,
    *,
    model: HuggingFaceModelAdapter,
    split: str,
    limit: Optional[int],
    max_new_tokens: int,
) -> Dict[str, object]:
    try:
        dataset = load_dataset("THUDM/LongBench", dataset_name, split=split)
    except Exception as exc:  # pragma: no cover - depends on external datasets
        return {"error": f"{type(exc).__name__}: {exc}"}

    references: List[List[str]] = []
    prompts: List[str] = []
    for sample in dataset:
        prompt = _build_prompt(sample)
        refs = _extract_references(sample)
        if not refs:
            continue
        prompts.append(prompt)
        references.append(refs)
        if limit and len(prompts) >= limit:
            break

    if not prompts:
        return {"error": "No evaluable samples"}

    predictions = _summarize(prompts, model, max_new_tokens)

    metrics: Dict[str, float] = {}
    if dataset_name in _ROUGE_TASKS:
        from evaluate import load as load_metric

        rouge = load_metric("rouge")
        rouge_scores = rouge.compute(predictions=predictions, references=[refs[0] for refs in references])
        metrics.update({f"rouge_{key}": float(val) for key, val in rouge_scores.items()})
    else:
        scores = [_f1_score(pred, refs) for pred, refs in zip(predictions, references)]
        metrics["f1"] = float(statistics.mean(scores))

    return {
        "num_samples": len(prompts),
        "metrics": metrics,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument("--bench", default=",".join(_DEFAULT_TASKS), help="Comma separated task list")
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--suite", default="long", help="Results suite name")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget per prompt")
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
    tasks = [task.strip() for task in args.bench.split(",") if task.strip()]
    if not tasks:
        raise ValueError("No tasks specified")

    model = _load_model(config, args)

    task_results: Dict[str, Mapping[str, object]] = {}
    f1_scores: List[float] = []
    rouge_scores: List[float] = []

    for task in tasks:
        result = _evaluate_task(
            task,
            model=model,
            split=args.split,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
        task_results[task] = result
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            if "f1" in metrics:
                f1_scores.append(float(metrics["f1"]))
            if "rouge_l" in metrics:
                rouge_scores.append(float(metrics["rouge_l"]))

    summary: Dict[str, object] = {"tasks": task_results}
    if f1_scores:
        summary["macro_f1"] = float(statistics.mean(f1_scores))
    if rouge_scores:
        summary["macro_rouge_l"] = float(statistics.mean(rouge_scores))

    metrics_path = suite_dir / "metrics.json"
    save_json(metrics_path, summary)
    print(f"Saved LongBench results to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
