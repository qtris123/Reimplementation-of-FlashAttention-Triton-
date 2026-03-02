"""
Triton kernel — non-causal FlashAttention with GQA support.

Adapted from problem_3.py + GQA head mapping from problem_5.py.
Single-pass over all KV blocks (no masking needed).
"""

import triton
import triton.language as tl


@triton.jit
def flash_noncausal_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Strides (B, H, S dimensions — D stride is always 1)
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Scalars
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Compile-time constants
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
):
    """
    Non-causal flash attention kernel.
    Iterates over ALL key blocks for each query block.
    Supports GQA via head-index remapping.
    """
    # --- Block / head identification ---
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA: map query head → shared KV head
    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # --- Accumulators ---
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # --- Load query block ---
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = (Q_ptr
              + batch_idx * q_stride_b
              + q_head_idx * q_stride_h
              + q_offsets[:, None] * q_stride_s
              + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    # Scale for exp2-based softmax: scale * log2(e)
    qk_scale = softmax_scale * 1.44269504

    # --- Main loop: all KV blocks ---
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K (transposed layout: HEAD_DIM × BLOCK_N)
        k_ptrs = (K_ptr
                  + batch_idx * k_stride_b
                  + kv_head_idx * k_stride_h
                  + k_offsets[None, :] * k_stride_s
                  + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V
        v_ptrs = (V_ptr
                  + batch_idx * v_stride_b
                  + kv_head_idx * v_stride_h
                  + k_offsets[:, None] * v_stride_s
                  + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Attention scores
        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Online softmax update
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        scale_factor = tl.exp2(m_i - m_new)
        acc *= scale_factor[:, None]
        l_i *= scale_factor

        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc += tl.dot(p_ij.to(v_block.type), v_block)
        l_i += tl.sum(p_ij, axis=1)

        m_i = m_new

    # --- Normalise and write output ---
    acc = acc / (l_i[:, None] + 1e-6)

    o_ptrs = (O_ptr
              + batch_idx * q_stride_b
              + q_head_idx * q_stride_h
              + q_offsets[:, None] * q_stride_s
              + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)
