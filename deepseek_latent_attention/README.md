# DeepSeek Latent Attention

DeepSeek Latent Attention is a research-oriented PyTorch project that reproduces and extends the Multi-Head Latent Attention (MLA) paradigm popularized by DeepSeek-V3. The repository offers dense and sparse latent attention layers, transformer building blocks, and lightweight training scaffolds so you can benchmark MLA against standard multi-head attention.

---

## Why Latent Attention?

Classical multi-head attention scales quadratically with the embedding dimension \(D\). Latent attention introduces a low-dimensional latent space \(d_\text{latent} \ll D\) to compute attention weights:

1. Project queries and keys into the latent space: \(Q_L = W_q Q\), \(K_L = W_k K\).
2. Form attention weights using \(\mathrm{softmax}\left(\frac{Q_L K_L^\top}{\sqrt{d_\text{latent}}}\right)\).
3. Aggregate values and project back with \(W_o\).

This reduces complexity from \(\mathcal{O}(D^2)\) to \(\mathcal{O}(D \cdot d_\text{latent})\) while preserving expressive capacity. Optional sparsity masks further prune attention scores, accelerating long-context workloads.

```
Q (B×T×D) --Wq--> Q_latent (B×H×T×d_l)
                          │
                          ├── softmax(Q_latent · K_latentᵀ / √d_l)
                          ▼
                  latent attention weights
                          │
V (B×T×D) --reshape--> values (B×H×T×d_h)
                          │
                          └── contract + Wo → output (B×T×D)
```

---

## Repository Layout

```
deepseek_latent_attention/
├── src/
│   ├── core/            # latent attention kernels, sparsity ops, fusion logic
│   ├── models/          # transformer block + config/registry helpers
│   ├── train/           # dataset loader, training loop, eval metrics
│   └── utils/           # logging & profiling utilities
├── tests/               # pytest suite covering dense/sparse correctness
├── notebooks/           # interactive demos & comparisons
├── README.md            # this file
├── LICENSE              # MIT License
└── requirements.txt     # minimal dependencies
```

---

## Key Features

- **Latent Attention Core** – Modular `LatentAttention` module with interpretable statistics and gradient-safe projections.
- **Sparse Compatibility** – Block-sparse and top-k masking utilities plus `LatentSparseAttention` for hybrid inference.
- **Fusion Layer** – Blend dense and sparse heads with learnable gating in `FusionLatentLayer`.
- **Training Scaffolding** – Configurable transformer blocks, dataset loader stub, and training/evaluation loop ready for experimentation.
- **Profiling Hooks** – FLOP/memory estimators and logging helpers to track per-head entropy, sparsity, and latency.

---

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.0 or newer
- Optional: CUDA toolkit for GPU experiments

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick Test

```bash
pytest
```

> **Note:** Tests assume a working PyTorch installation. CPU-only environments will run but may be slower for long sequences.

---

## Usage Example

```python
import torch
from deepseek_latent_attention.src.core.mha_latent import LatentAttention

batch, seq_len, embed_dim = 2, 128, 512
inputs = torch.randn(batch, seq_len, embed_dim)

attn = LatentAttention(
    embed_dim=embed_dim,
    num_heads=8,
    latent_dim_ratio=0.25,
    track_stats=True,
)

output, weights, stats = attn(inputs, inputs, inputs, need_weights=True)
print(output.shape)   # torch.Size([2, 128, 512])
print(stats["entropy"].shape)  # per-head entropy diagnostics
```

### Hybrid Sparse Variant

```python
from deepseek_latent_attention.src.core.mha_latent import LatentSparseAttention
from deepseek_latent_attention.src.core.sparse_utils import TopKMaskBuilder

mask_builder = TopKMaskBuilder(k=32)
attn = LatentSparseAttention(
    embed_dim=embed_dim,
    num_heads=8,
    latent_dim_ratio=0.25,
    mask_builder=mask_builder,
)
output, _, sparse_stats = attn(inputs, inputs, inputs)
```

---

## Configuration & Training

High-level configuration is centralized in `src/models/config.py`. Important flags include:

- `latent_dim_ratio`: fraction of the embedding dimension allocated to the latent space.
- `use_sparse`: toggles sparse masking in the transformer block.
- `sparse_topk` / `sparse_block`: choose sparsity pattern and parameters.
- `log_attention_stats`: enables GEPA Thought-Trace style logging hooks.

You can instantiate a transformer block via the registry:

```python
from deepseek_latent_attention.src.models.registry import build_model
model = build_model("latent_transformer", config_overrides={"num_layers": 4})
```

Training utilities live in `src/train/`. The `train_loop.py` file exposes a `train_epoch` function that accepts the config, dataloaders, and optional profiler callbacks.

---

## Profiling & Evaluation

- `src/utils/profiling.py` contains FLOP/memory estimators for dense vs. latent attention.
- `src/train/eval_metrics.py` exposes accuracy, perplexity, and sparsity-aware metrics.
- The `logging` module captures per-head entropy and sparsity for interpretability studies.

---

## Notebooks

- **`demo_latent_attention.ipynb`** – Walk-through of latent attention mechanics with visualization hooks.
- **`compare_sparse_dense.ipynb`** – Benchmarks dense vs. sparse latent attention on synthetic workloads.

Open them with Jupyter or VS Code to experiment interactively.

---

## Testing

The `tests/test_mha_latent.py` suite validates:

- Dense equivalence when the sparse mask is fully open.
- Parameter count and gradient propagation through latent projections.
- Correct application of block and top-k masks.

Run `pytest -k latent` to execute only latent-attention related tests.

---

## Citation

If you use this repository in your research, please cite the original DeepSeek-V3 work:

> DeepSeek-V3 Team. *DeepSeek-V3: Scaling Latent Attention* (2024).

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

