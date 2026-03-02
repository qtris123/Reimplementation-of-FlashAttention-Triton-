"""
Triton backend dispatcher — picks the right kernel and launches it.

Routes to either the non-causal or the causal kernel based on config,
handles stride computation and grid sizing.
"""

import math
import torch
import triton

from ..config import AttentionConfig
from ..utils import compute_softmax_scale
from .kernels.noncausal import flash_noncausal_kernel
from .kernels.causal import flash_causal_kernel


def triton_flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """
    Launch the appropriate Triton kernel for the given config.

    Args:
        Q: (B, H_q, N, D) on CUDA
        K: (B, H_kv, N, D) on CUDA
        V: (B, H_kv, N, D) on CUDA
        config: AttentionConfig

    Returns:
        O: (B, H_q, N, D) — attention output
    """
    batch, n_q_heads, seq_len, head_dim = Q.shape
    n_kv_heads = K.shape[1]

    O = torch.empty_like(Q)
    softmax_scale = compute_softmax_scale(head_dim)

    BLOCK_M = config.block_m
    BLOCK_N = config.block_n
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    if not config.is_causal:
        # ---- Non-causal path ----
        flash_noncausal_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    else:
        # ---- Causal path (supports GQA + SWA + Sinks) ----
        # Default window = full sequence when SWA is off
        window_size = config.window_size if config.window_size is not None else seq_len
        sink_size = config.sink_size

        flash_causal_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    return O
