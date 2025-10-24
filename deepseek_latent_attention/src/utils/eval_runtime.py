"""Runtime helpers shared across evaluation scripts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelLoadConfig:
    """Configuration for loading a Hugging Face causal LM."""

    model_name: str
    tokenizer_name: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    revision: str | None = None
    max_length: int = 4096


def parse_dtype(name: str) -> torch.dtype:
    """Map string identifiers to :mod:`torch` dtypes."""

    normalized = name.lower().replace("torch.", "")
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
    }
    if normalized not in mapping:
        raise KeyError(f"Unsupported dtype: {name}")
    return mapping[normalized]


class HuggingFaceModelAdapter:
    """Thin wrapper adding convenience helpers for inference."""

    def __init__(self, config: ModelLoadConfig) -> None:
        self.config = config
        dtype = parse_dtype(config.dtype)
        tokenizer_name = config.tokenizer_name or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=config.revision,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            revision=config.revision,
            torch_dtype=dtype,
        )
        self.model.to(config.device)
        self.model.eval()
        self.max_length = config.max_length

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> List[str]:
        """Generate continuations for a batch of prompts."""

        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        outputs = self.model.generate(
            **encoded,
            do_sample=do_sample or temperature > 0,
            temperature=max(temperature, 1e-5),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts

    def compute_perplexity(
        self, texts: Sequence[str], *, batch_size: int = 4
    ) -> float:
        """Compute perplexity using masked language modelling loss."""

        log_likelihoods: List[float] = []
        total_tokens = 0
        for index in range(0, len(texts), batch_size):
            batch = list(texts[index : index + batch_size])
            encodings = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            input_ids = encodings.input_ids
            target_ids = input_ids.clone()
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                mask = target_ids.ne(-100)
                if mask.sum() == 0:
                    continue
                negative_log_likelihood = outputs.loss * mask.sum()
            log_likelihoods.append(float(negative_log_likelihood.detach()))
            total_tokens += int(mask.sum())
        if total_tokens == 0:
            return float("nan")
        ppl = math.exp(sum(log_likelihoods) / total_tokens)
        return float(ppl)

    def score_choices(
        self,
        prompts: Sequence[str],
        choices: Sequence[Sequence[str]],
        *,
        batch_size: int = 1,
    ) -> List[int]:
        """Return the argmax choice index for each prompt."""

        winners: List[int] = []
        for start in range(0, len(prompts), batch_size):
            prompt_batch = list(prompts[start : start + batch_size])
            choice_batch = choices[start : start + batch_size]
            for prompt, prompt_choices in zip(prompt_batch, choice_batch):
                choice_scores = []
                for candidate in prompt_choices:
                    text = prompt + candidate
                    enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        log_prob = self.model(
                            **enc,
                            labels=enc.input_ids,
                        ).loss.item()
                    choice_scores.append(-log_prob)
                winners.append(int(np.argmax(choice_scores)))
        return winners


def prepare_results_dir(root: str | Path, model_tag: str, suite: str) -> Path:
    """Create the directory for a suite and return the resolved path."""

    target = Path(root) / model_tag / suite
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    """Persist a JSON payload with deterministic formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


__all__ = [
    "HuggingFaceModelAdapter",
    "ModelLoadConfig",
    "parse_dtype",
    "prepare_results_dir",
    "save_json",
]
