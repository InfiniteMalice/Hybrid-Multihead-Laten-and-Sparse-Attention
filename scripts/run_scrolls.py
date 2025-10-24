#!/usr/bin/env python3
"""Run SCROLLS long-context benchmarks (GovReport, NarrativeQA, QMSum)."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets import load_dataset

from deepseek_latent_attention.src.utils.config_loader import load_config, resolve_model_tag
from deepseek_latent_attention.src.utils.eval_runtime import (
    HuggingFaceModelAdapter,
    ModelLoadConfig,
    prepare_results_dir,
    save_json,
)

_SUMMARY_DATASETS: Dict[str, Tuple[str, Optional[str]]] = {
    "gov_report": ("ccdv/gov_report", None),
    "qmsum": ("EdinburghNLP/qmsum", None),
}

_QA_DATASETS: Dict[str, Tuple[str, Optional[str]]] = {
    "narrativeqa": ("narrativeqa", "plain_text"),
}


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


def _load_model(config: Mapping[str, object], overrides: argparse.Namespace) -> HuggingFaceModelAdapter:
    runtime = config.get("runtime", {})
    model_name = overrides.model or runtime.get("model_name_or_path")
    if model_name is None:
        raise ValueError("model_name_or_path must be specified")
    load_cfg = ModelLoadConfig(
        model_name=str(model_name),
        tokenizer_name=str(overrides.tokenizer or runtime.get("tokenizer_name_or_path") or model_name),
        device=str(overrides.device or runtime.get("device", "cuda")),
        dtype=str(overrides.dtype or runtime.get("dtype", "bfloat16")),
        revision=overrides.revision or runtime.get("revision"),
        max_length=int(overrides.max_length or runtime.get("max_length", 4096)),
    )
    return HuggingFaceModelAdapter(load_cfg)


def _summarize(
    dataset_name: str,
    *,
    model: HuggingFaceModelAdapter,
    split: str,
    limit: Optional[int],
    max_new_tokens: int,
) -> Dict[str, object]:
    hf_name, config_name = _SUMMARY_DATASETS[dataset_name]
    try:
        dataset = load_dataset(hf_name, config_name, split=split)
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}

    prompts: List[str] = []
    references: List[str] = []
    for sample in dataset:
        document = sample.get("document") or sample.get("dialogue") or sample.get("text")
        summary = sample.get("summary")
        if isinstance(document, Mapping):
            document = document.get("text") or document.get("article")
        if isinstance(summary, Mapping):
            summary = summary.get("text")
        if not document or not summary:
            continue
        prompts.append(str(document))
        references.append(str(summary))
        if limit and len(prompts) >= limit:
            break

    if not prompts:
        return {"error": "No evaluable samples"}

    generations = model.generate(prompts, max_new_tokens=max_new_tokens)

    from evaluate import load as load_metric

    rouge = load_metric("rouge")
    scores = rouge.compute(predictions=generations, references=references)
    metrics = {f"rouge_{key}": float(val) for key, val in scores.items()}
    return {"metrics": metrics, "num_samples": len(generations)}


def _qa(
    dataset_name: str,
    *,
    model: HuggingFaceModelAdapter,
    split: str,
    limit: Optional[int],
    max_new_tokens: int,
) -> Dict[str, object]:
    hf_name, config_name = _QA_DATASETS[dataset_name]
    try:
        dataset = load_dataset(hf_name, config_name, split=split)
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}

    prompts: List[str] = []
    references: List[List[str]] = []
    for sample in dataset:
        if "document" in sample and isinstance(sample["document"], Mapping):
            context = sample["document"].get("text") or sample["document"].get("summary")
        else:
            context = sample.get("context")
        question = sample.get("question")
        if isinstance(question, Mapping):
            question = question.get("text")
        answers = sample.get("answers") or sample.get("answer")
        if isinstance(answers, Mapping):
            answers = answers.get("text") or answers.get("answers")
        if isinstance(answers, str):
            answers = [answers]
        if not context or not question or not answers:
            continue
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        prompts.append(prompt)
        references.append([str(ans) for ans in answers])
        if limit and len(prompts) >= limit:
            break

    if not prompts:
        return {"error": "No evaluable samples"}

    predictions = model.generate(prompts, max_new_tokens=max_new_tokens)
    scores = [_f1_score(pred, refs) for pred, refs in zip(predictions, references)]
    metrics = {"f1": float(statistics.mean(scores))}
    return {"metrics": metrics, "num_samples": len(predictions)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument(
        "--tasks",
        default="gov_report,narrativeqa,qmsum",
        help="Comma separated SCROLLS task list",
    )
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--suite", default="scrolls", help="Results suite name")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per task")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget")
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
    rouge_scores: List[float] = []
    f1_scores: List[float] = []

    for task in tasks:
        if task in _SUMMARY_DATASETS:
            result = _summarize(
                task,
                model=model,
                split=args.split,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
            )
        elif task in _QA_DATASETS:
            result = _qa(
                task,
                model=model,
                split=args.split,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            summary["tasks"][task] = {"error": "Unknown SCROLLS task"}
            continue

        summary["tasks"][task] = result
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            if "f1" in metrics:
                f1_scores.append(float(metrics["f1"]))
            if "rouge_l" in metrics:
                rouge_scores.append(float(metrics["rouge_l"]))

    if rouge_scores:
        summary["macro_rouge_l"] = float(statistics.mean(rouge_scores))
    if f1_scores:
        summary["macro_f1"] = float(statistics.mean(f1_scores))

    metrics_path = suite_dir / "metrics.json"
    save_json(metrics_path, summary)
    print(f"Saved SCROLLS results to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
