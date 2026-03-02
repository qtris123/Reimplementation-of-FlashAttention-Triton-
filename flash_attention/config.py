"""
Attention configuration — single dataclass controls all behavior.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttentionConfig:
    """
    Configuration for the unified FlashAttention module.

    Attributes:
        backend:     "pytorch" (tiled, from problem 1) or "triton" (GPU kernels, problems 3-7).
        is_causal:   Whether to apply causal (autoregressive) masking.
        num_kv_heads: Number of key/value heads for GQA. None = same as query heads (MHA).
        window_size: Sliding-window size. None = full attention.
        sink_size:   Number of initial "sink" tokens always attended to. 0 = disabled.
        block_m:     Query tile size for tiling / kernel launch.
        block_n:     Key/Value tile size for tiling / kernel launch.
    """

    backend: str = "triton"
    is_causal: bool = True
    num_kv_heads: Optional[int] = None
    window_size: Optional[int] = None
    sink_size: int = 0
    block_m: int = 128
    block_n: int = 64

    def __post_init__(self):
        if self.backend not in ("pytorch", "triton"):
            raise ValueError(f"backend must be 'pytorch' or 'triton', got '{self.backend}'")
        if self.sink_size > 0 and self.window_size is None:
            raise ValueError("sink_size > 0 requires window_size to be set")
        if self.window_size is not None and not self.is_causal:
            raise ValueError("Sliding window attention requires is_causal=True")

    @property
    def use_gqa(self) -> bool:
        """True if grouped-query attention is active."""
        return self.num_kv_heads is not None

    @property
    def use_sliding_window(self) -> bool:
        """True if sliding-window attention is active."""
        return self.window_size is not None

    @property
    def use_sinks(self) -> bool:
        """True if attention sinks are active."""
        return self.sink_size > 0
