"""Top-level package for DeepSeek-style latent attention modules."""

from .core.mha_latent import LatentAttention, LatentSparseAttention

__all__ = [
    "LatentAttention",
    "LatentSparseAttention",
]

try:  # pragma: no cover - optional dependency on einops
    from .core.config import AttnConfig
    from .core.hybrid_attention import MultiheadAttn
except ModuleNotFoundError:  # pragma: no cover - handled when optional deps missing
    AttnConfig = None  # type: ignore[assignment]
    MultiheadAttn = None  # type: ignore[assignment]
else:  # pragma: no cover - trivial container
    __all__.extend(["AttnConfig", "MultiheadAttn"])
