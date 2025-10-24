# Self-Anchor Attention

Self-Anchor adds a configurable bias to attention logits after score computation and
before the softmax. The bias is derived from `detect_anchors`, which supports:

- `cot_markers`: identifies structured reasoning markers such as `<cot>` or answer
  delimiters, and selects low-index numeric tokens.
- `regex`: selects punctuation-heavy and enumerated tokens through lightweight
  regular expressions.
- `saliency_topk`: marks the top tokens according to a supplied saliency score.

The module exposes `SelfAnchorConfig` and `apply_self_anchor`. When enabled the bias
is broadcast across heads and queries, nudging probability mass towards semantic
anchors without altering baseline masking semantics.
