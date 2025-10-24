#!/usr/bin/env python3
"""Generate diagnostic stats (gate, ssm_norm, entropy, spikes) for hybrid modules."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from deepseek_latent_attention.src.core.hybrid_mamba_attn import HybridMambaAttention, HybridMambaConfig
from deepseek_latent_attention.src.core.mamba_anchor import MambaAnchorConfig
from deepseek_latent_attention.src.core.mamba_block import MambaBlock, MambaConfig
from deepseek_latent_attention.src.core.mha_latent import LatentAttention, LatentSparseAttention
from deepseek_latent_attention.src.core.self_anchor import SelfAnchorConfig
from deepseek_latent_attention.src.utils.config_loader import load_config, resolve_model_tag
from deepseek_latent_attention.src.utils.eval_runtime import prepare_results_dir


def _build_attention(config: Mapping[str, object]) -> HybridMambaAttention:
    model_cfg = config.get("model", {})
    d_model = int(model_cfg.get("d_model", 256))
    n_heads = int(model_cfg.get("n_heads", 8))
    latent_ratio = float(model_cfg.get("latent_ratio", 0.25))
    dropout = float(model_cfg.get("attn_dropout", 0.0))

    base_type = str(model_cfg.get("base", "MLA_PLUS_SPARSE")).upper()
    attn_cls = LatentSparseAttention if "SPARSE" in base_type else LatentAttention
    base_attn = attn_cls(
        embed_dim=d_model,
        num_heads=n_heads,
        latent_dim_ratio=latent_ratio,
        dropout=dropout,
        track_stats=True,
    )
    base_attn.d_model = d_model  # attribute required by HybridMambaAttention

    mamba_cfg = config.get("mamba", {})
    mode = str(mamba_cfg.get("mode", "OFF")).upper()
    hybrid_cfg = HybridMambaConfig(enable=mode != "OFF", apply_to=mode.lower(), gate_bias=0.0)

    self_anchor_cfg: Optional[SelfAnchorConfig] = None
    self_anchor_dict = config.get("self_anchor", {})
    if self_anchor_dict.get("enable"):
        self_anchor_cfg = SelfAnchorConfig(**self_anchor_dict)

    mamba_anchor_cfg: Optional[MambaAnchorConfig] = None
    mamba_anchor_dict = config.get("mamba_anchor", {})
    if mamba_anchor_dict.get("enable"):
        mamba_anchor_cfg = MambaAnchorConfig(**mamba_anchor_dict)

    if hybrid_cfg.enable:
        mamba_block = MambaBlock(
            MambaConfig(
                d_model=d_model,
                state_size=int(mamba_cfg.get("state_size", 64)),
                dt_rank=int(mamba_cfg.get("dt_rank", 16)),
                conv_kernel_size=int(mamba_cfg.get("conv_kernel_size", 4)),
                dropout=float(mamba_cfg.get("dropout", 0.0)),
            )
        )
    else:
        mamba_block = None

    return HybridMambaAttention(
        base_attn,
        mamba_block,
        config=hybrid_cfg,
        self_anchor=self_anchor_cfg,
        mamba_anchor=mamba_anchor_cfg,
    )


def _collect_stats(
    module: HybridMambaAttention,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    module.to(device)
    module.eval()
    with torch.no_grad():
        inputs = torch.randn(batch_size, seq_len, module.base_attn.embed_dim, device=device)
        tokens = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        _, stats = module(
            inputs,
            tokens=tokens,
            return_stats=True,
        )
    gate = stats.get("gate")
    ssm_norm = stats.get("ssm_norm")
    entropy = stats.get("attn_entropy")
    spikes = stats.get("meta_token_indices")

    gate_np = gate.detach().cpu().numpy() if gate is not None else np.empty(0)
    ssm_np = ssm_norm.detach().cpu().numpy() if ssm_norm is not None else np.empty(0)
    entropy_np = entropy.detach().cpu().numpy() if entropy is not None else np.empty(0)

    if spikes is not None and spikes.numel() > 0:
        spike_mask = torch.zeros(gate.shape[:2], dtype=torch.bool)
        spike_mask[spikes[:, 0], spikes[:, 1]] = True
        spikes_np = spike_mask.cpu().numpy()
    else:
        spikes_np = (gate_np > 0.9) if gate_np.size else np.empty(0)

    return {
        "gate": gate_np,
        "ssm_norm": ssm_np,
        "attn_entropy": entropy_np,
        "spikes": spikes_np,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Experiment YAML config path")
    parser.add_argument("--output-dir", default="results", help="Root directory for outputs")
    parser.add_argument("--suite", default="stats", help="Output suite name")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers to simulate")
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--device", help="Override device")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.cfg)
    cfg_path = Path(args.cfg)
    model_tag = resolve_model_tag(config, cfg_path.stem)
    suite_dir = prepare_results_dir(args.output_dir, model_tag, args.suite)

    runtime = config.get("runtime", {})
    seq_len = args.seq_len or int(runtime.get("seq_len", 128))
    device_str = args.device or runtime.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    module = _build_attention(config)

    for layer in range(args.layers):
        stats = _collect_stats(
            module,
            seq_len=seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        stats_path = suite_dir / f"{run_id}_{layer}.npz"
        np.savez(stats_path, **stats)
        print(f"Saved stats to {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
