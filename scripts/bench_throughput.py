#!/usr/bin/env python3
"""Benchmark latency, throughput, and memory usage for causal LMs."""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from deepseek_latent_attention.src.utils.config_loader import load_config, resolve_model_tag
from deepseek_latent_attention.src.utils.eval_runtime import (
    HuggingFaceModelAdapter,
    ModelLoadConfig,
    prepare_results_dir,
)


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


def _build_inputs(model: HuggingFaceModelAdapter, seq_len: int, batch_size: int) -> Dict[str, torch.Tensor]:
    token = model.tokenizer.eos_token or " "
    prompt = token * max(1, seq_len - 1)
    encoded = model.tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    ).to(model.device)
    return {
        "input_ids": encoded.input_ids,
        "attention_mask": encoded.attention_mask,
    }


def _kv_bytes_per_token(model: HuggingFaceModelAdapter) -> float:
    config = getattr(model.model, "config", None)
    if config is None:
        return float("nan")
    hidden = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    heads = getattr(config, "num_attention_heads", None)
    if hidden is None or layers is None:
        return float("nan")
    dtype_size = torch.tensor([], dtype=model.model.dtype).element_size()
    kv_per_layer = hidden * 2  # key and value per token
    if heads is not None and heads > 0:
        kv_per_layer = hidden * 2
    total = kv_per_layer * layers * dtype_size
    return float(total)


def _benchmark_case(
    model: HuggingFaceModelAdapter,
    seq_len: int,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    inputs = _build_inputs(model, seq_len, batch_size)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    device = model.device
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.model(input_ids=input_ids, attention_mask=attention_mask)

    elapsed: List[float] = []
    for _ in range(iterations):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            model.model(input_ids=input_ids, attention_mask=attention_mask)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - start)

    avg_time = float(np.mean(elapsed))
    tokens = seq_len * batch_size
    tokens_per_second = tokens / avg_time if avg_time > 0 else float("nan")
    ms_per_token = 1000.0 / tokens_per_second if tokens_per_second else float("nan")
    latency_ms = avg_time * 1000.0
    peak_vram = 0.0
    if is_cuda:
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "tokens_per_second": float(tokens_per_second),
        "ms_per_token": float(ms_per_token),
        "latency_ms": float(latency_ms),
        "peak_vram_mb": float(peak_vram),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument("--seq", required=True, help="Comma separated sequence lengths")
    parser.add_argument("--bs", required=True, help="Comma separated batch sizes")
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--suite", default="perf", help="Output suite name")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--tokenizer", help="Override tokenizer name")
    parser.add_argument("--dtype", help="Override dtype")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--revision", help="Model revision identifier")
    parser.add_argument("--max-length", type=int, help="Tokenizer maximum length")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    cfg_path = Path(args.cfg)
    model_tag = resolve_model_tag(config, cfg_path.stem)
    suite_dir = prepare_results_dir(args.output_dir, model_tag, args.suite)

    model = _load_model(config, args)
    kv_bytes = _kv_bytes_per_token(model)

    seq_lengths = [int(value) for value in args.seq.split(",") if value.strip()]
    batch_sizes = [int(value) for value in args.bs.split(",") if value.strip()]

    records: List[Dict[str, float]] = []
    for seq_len, batch_size in product(seq_lengths, batch_sizes):
        stats = _benchmark_case(
            model,
            seq_len=seq_len,
            batch_size=batch_size,
            warmup=args.warmup,
            iterations=args.iters,
        )
        stats["kv_bytes_per_token"] = kv_bytes
        records.append(stats)

    frame = pd.DataFrame.from_records(records)
    output_path = suite_dir / "latency_mem.csv"
    frame.to_csv(output_path, index=False)
    print(f"Saved latency benchmark to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
