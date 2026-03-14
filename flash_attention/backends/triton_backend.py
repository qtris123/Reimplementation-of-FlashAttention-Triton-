"""
Triton backend dispatcher — picks the right kernel and launches it.

Routes to the appropriate kernel based on config and input shape:
  - Non-causal → flash_noncausal_kernel
  - Causal + prefill (Q_seq == K_seq) → flash_causal_kernel
  - Causal + decode (Q_seq == 1) → flash_decode_kernel
"""

import math
import torch
import triton

from ..config import AttentionConfig
from ..utils import compute_softmax_scale
from .kernels.noncausal import flash_noncausal_kernel
from .kernels.causal import flash_causal_kernel
from .kernels.decode import flash_decode_kernel


def triton_flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: AttentionConfig,
    q_pos_offset: int = 0,
) -> torch.Tensor:
    """
    Launch the appropriate Triton kernel for the given config.

    Args:
        Q: (B, H_q, N_q, D) on CUDA
        K: (B, H_kv, N_k, D) on CUDA
        V: (B, H_kv, N_k, D) on CUDA
        config: AttentionConfig
        q_pos_offset: Absolute position of Q[0] (for decode)

    Returns:
        O: (B, H_q, N_q, D) — attention output
    """
    batch, n_q_heads, q_seq_len, head_dim = Q.shape
    n_kv_heads = K.shape[1]
    k_seq_len = K.shape[2]

    O = torch.empty_like(Q)
    softmax_scale = compute_softmax_scale(head_dim)

    BLOCK_M = config.block_m
    BLOCK_N = config.block_n

    if not config.is_causal:
        # ---- Non-causal path ----
        grid = (triton.cdiv(q_seq_len, BLOCK_M), batch * n_q_heads)
        flash_noncausal_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            softmax_scale,
            q_seq_len,  # for non-causal, Q and K should be same
            n_q_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    elif q_seq_len == 1:
        # ---- Decode path (single-token Q) ----
        window_size = config.window_size if config.window_size is not None else k_seq_len
        sink_size = config.sink_size

        grid = (1, batch * n_q_heads)
        flash_decode_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            softmax_scale,
            q_pos_offset,   # absolute position of the query token
            k_seq_len,      # total K/V length
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_N=BLOCK_N,
        )
    else:
        # ---- Prefill path (Q_seq == K_seq, uses existing causal kernel) ----
        window_size = config.window_size if config.window_size is not None else q_seq_len
        sink_size = config.sink_size

        grid = (triton.cdiv(q_seq_len, BLOCK_M), batch * n_q_heads)
        flash_causal_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            softmax_scale,
            q_seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    return O
