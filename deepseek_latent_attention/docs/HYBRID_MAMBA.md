# Hybrid Mamba Attention

`HybridMambaAttention` fuses the existing attention pathway with a lightweight
selective-scan state space module. The fusion operates per token with a learned
gate:

```
y_attn = base_attn(x, mask)
y_ssm = mamba(x)
gate = sigmoid(W_g x)
y = gate * y_ssm + (1 - gate) * y_attn
```

When `return_stats=True` the module exports diagnostic tensors (`gate`, `ssm_norm`,
`attn_entropy`, and `meta_token_indices`) to simplify experiment logging. The
hybrid path is toggled via `HybridMambaConfig.enable` and scoped to MLA, Sparse, or
both pathways using `HybridMambaConfig.apply_to`.
