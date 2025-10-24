# Mamba Anchor

`MambaAnchor` nudges the hybrid gate towards anchor tokens detected by
`detect_anchors`. The operation is simple and bounded:

```
gate = clamp(gate + alpha * anchor_mask, 0, 1)
```

`MambaAnchorConfig` controls the strength (`alpha`) and discovery strategy. The
module is designed to be orthogonal to self-anchored attention; enabling
`MambaAnchor` does not modify the attention logits, only the fusion gate.
