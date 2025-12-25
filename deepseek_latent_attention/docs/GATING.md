# Attention Gating

This repository supports an optional gating mechanism inspired by the NeurIPS 2025 work on
integrating gating into standard softmax attention to improve stability and long-context
robustness. Gating introduces learnable, differentiable scalars that scale attention scores
before the softmax, allowing light fine-tuning without retraining the base model.

## Where Gating Sits

Gating is applied **after** attention scores are computed and masks are applied, but
**before** the softmax:

```text
scores = compute_scores(q, k)
scores = apply_mask(scores)
scores = gating(q, k, scores)  # optional
attn = softmax(scores)
```

## Methods

- **HEADWISE**: a single gate per head. Gates are broadcast to `[1, H, 1, 1]` and scale each
  head’s scores uniformly.
- **TOKENWISE**: a per-head projection from query vectors to scalar gates for each query
  position (`[B, H, L_q, 1]`), enabling token-specific scaling.

Both methods are no-ops when `method: none`.

## Configuration Example

```yaml
model:
  base: MLA_ONLY
  gating:
    method: headwise
    init_bias: 0.0
    max_scale: 1.5
    dropout: 0.0
```

## Recommended Workflow

1. Start with gating disabled (default behavior).
2. Enable **HEADWISE** gating and fine-tune lightly on existing tasks.
3. Explore **TOKENWISE** gating for long-context or stability-sensitive workloads.
