# Agent Guidelines

## Global Requirements
- Enforce a maximum line length of 100 characters for all source files.
- Organize imports with the standard library, third-party, then local sections.
- Group all PyTorch imports together and include `from torch import Tensor` when type hints are needed.
- Avoid unused imports, especially from `torch.nn.functional` and `einops`.
- Apply Black formatting with `--line-length 100` and ensure 4-space indentation and trailing commas.
- Document tensor shapes in attention modules and assert divisibility checks such as `d_model % n_heads == 0`.
- Break long tensor operations into multiple steps when needed to respect the line-length rule.

## Attention Module Conventions
- Follow the provided multi-head attention template structure when implementing new modules.
- Use standard abbreviations: attn, mhead, dim, seq, q, k, v, sc, probs.
- Handle attention score scaling explicitly and apply masks with `masked_fill` semantics.

These rules apply to the entire repository unless superseded by a nested `AGENTS.md`.
