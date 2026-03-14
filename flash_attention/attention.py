"""
FlashAttention module — the main user-facing API.

Wraps both backends behind a single nn.Module interface.
Switch between PyTorch (reference) and Triton (high-performance)
via config or at runtime with ``switch_backend()``.
"""

import torch
import torch.nn as nn

from .config import AttentionConfig
from .utils import validate_qkv_shapes
from .backends.pytorch import pytorch_flash_attention_forward
from .backends.triton_backend import triton_flash_attention_forward


class FlashAttention(nn.Module):
    """
    Unified FlashAttention module.

    Supports:
      - Causal / non-causal masking
      - Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)
      - Sliding Window Attention (SWA)
      - Attention Sinks
      - Two switchable backends: ``"pytorch"`` and ``"triton"``

    Example::

        config = AttentionConfig(
            backend="triton",
            is_causal=True,
            num_kv_heads=2,
            window_size=256,
            sink_size=4,
        )
        attn = FlashAttention(config).cuda()
        output = attn(Q, K, V)  # (B, H_q, N, D)

        # Switch to PyTorch backend for debugging
        attn.switch_backend("pytorch")
        output_ref = attn(Q, K, V)
    """

    def __init__(self, config: AttentionConfig | None = None):
        super().__init__()
        self.config = config or AttentionConfig()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, q_pos_offset: int = 0) -> torch.Tensor:
        """
        Compute attention.

        Args:
            Q: (B, H_q, N_q, D) — queries (N_q can be 1 during decode)
            K: (B, H_kv, N_k, D) — keys (full context with KV cache)
            V: (B, H_kv, N_k, D) — values
            q_pos_offset: Absolute position of Q[0] in the full sequence.
                          For prefill: 0. For decode at step t: t.

        Returns:
            O: (B, H_q, N_q, D)
        """
        validate_qkv_shapes(Q, K, V, self.config.num_kv_heads)

        if self.config.backend == "pytorch":
            return pytorch_flash_attention_forward(Q, K, V, self.config, q_pos_offset=q_pos_offset)
        else:
            return triton_flash_attention_forward(Q, K, V, self.config, q_pos_offset=q_pos_offset)

    def switch_backend(self, backend: str):
        """
        Swap backend at runtime.

        Args:
            backend: ``"pytorch"`` or ``"triton"``
        """
        if backend not in ("pytorch", "triton"):
            raise ValueError(f"backend must be 'pytorch' or 'triton', got '{backend}'")
        self.config.backend = backend

    def __repr__(self):
        parts = [
            f"backend={self.config.backend!r}",
            f"causal={self.config.is_causal}",
        ]
        if self.config.use_gqa:
            parts.append(f"kv_heads={self.config.num_kv_heads}")
        if self.config.use_sliding_window:
            parts.append(f"window={self.config.window_size}")
        if self.config.use_sinks:
            parts.append(f"sinks={self.config.sink_size}")
        return f"FlashAttention({', '.join(parts)})"
