# DeepSeek Latent Attention

DeepSeek-V3 style Multi-Head Latent Attention (MLA) layers implemented in PyTorch.
The project provides dense and sparse latent attention variants, transformer
blocks, and light-weight training scaffolding for experimentation.

## Key Idea

Latent attention projects the high-dimensional query/key representations to a
compact latent space before computing attention scores, reducing the quadratic
complexity from :math:`O(D^2)` to :math:`O(D \cdot d_\text{latent})` where
``d_latent << D``.

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

Sparse extensions support block and top-k masking, enabling the hybrid
LatentSparseAttention module.

## Repository Layout

- `src/core/`: core attention kernels, sparsity utilities, normalization and fusion layers.
- `src/models/`: transformer block, configs, and registry utilities.
- `src/train/`: toy dataset loader, train loop, metrics, and benchmarking helpers.
- `tests/`: PyTest unit tests verifying dense equivalence, gradients, and sparsity masks.
- `notebooks/`: starter notebooks for MLA demonstrations.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

### Example Usage

```python
import torch
from deepseek_latent_attention.src.core.mha_latent import LatentAttention

attn = LatentAttention(embed_dim=512, num_heads=8, latent_dim_ratio=0.25)
inputs = torch.randn(2, 128, 512)
out, weights, stats = attn(inputs, inputs, inputs, need_weights=True)
print(out.shape)
```

## Citation

Please cite the DeepSeek-V3 work and the MLA concept when using this project in
your research:

> DeepSeek-V3 Team. *DeepSeek-V3: Scaling Latent Attention* (2024).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
