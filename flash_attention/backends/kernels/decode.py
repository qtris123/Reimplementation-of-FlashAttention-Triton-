"""
Triton kernel — single-query decode attention (Q_seq == 1).

Optimised for the autoregressive decode step where Q has 1 token
but K/V span the full context. Much simpler than the prefill kernel:
  - No tiling over Q (single row)
  - Single pass over all K/V blocks within the causal window
  - Supports GQA, Sliding Window, and Attention Sinks
"""

import triton
import triton.language as tl


@triton.jit
def flash_decode_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Strides (B, H, S dims — D stride is 1)
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Scalars
    softmax_scale,
    Q_POS,          # absolute position of the query token
    K_SEQ_LEN,      # total K/V sequence length (cache size)
    N_Q_HEADS,
    N_KV_HEADS,
    # Compile-time constants
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE:   tl.constexpr,
    HEAD_DIM:    tl.constexpr,
    BLOCK_N:     tl.constexpr,
):
    """
    Decode kernel: Q has a SINGLE token at absolute position Q_POS,
    K/V have K_SEQ_LEN tokens (0 .. K_SEQ_LEN-1).

    Grid: (1, batch * n_q_heads)
    """
    # ================================================================
    # 1. Identify batch and head
    # ================================================================
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA head mapping
    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # ================================================================
    # 2. Load single query row (1, HEAD_DIM)
    # ================================================================
    q_ptrs = (Q_ptr
              + batch_idx * q_stride_b
              + q_head_idx * q_stride_h
              + tl.arange(0, HEAD_DIM))
    q_row = tl.load(q_ptrs).to(tl.float32)  # (HEAD_DIM,)

    qk_scale = softmax_scale * 1.44269504

    # ================================================================
    # 3. Accumulators
    # ================================================================
    m_i = tl.full([1], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # ================================================================
    # 4. Compute the range of K positions to attend to
    # ================================================================
    # Causal: attend to k <= Q_POS
    # Window: attend to k >= Q_POS - WINDOW_SIZE + 1
    # Sinks: always attend to k < SINK_SIZE

    causal_end = tl.minimum(Q_POS + 1, K_SEQ_LEN)

    # Window start (excluding sinks, handled separately)
    win_left = Q_POS - (WINDOW_SIZE - 1)
    window_start = tl.maximum(0, win_left)

    # ================================================================
    # Phase 0: Sink tokens (if any)
    # ================================================================
    if SINK_SIZE > 0:
        for start_n in range(0, SINK_SIZE, BLOCK_N):
            k_offsets = start_n + tl.arange(0, BLOCK_N)

            # Load K block: (HEAD_DIM, BLOCK_N)
            k_ptrs = (K_ptr
                      + batch_idx * k_stride_b
                      + kv_head_idx * k_stride_h
                      + k_offsets[None, :] * k_stride_s
                      + tl.arange(0, HEAD_DIM)[:, None])
            k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < K_SEQ_LEN, other=0.0).to(tl.float32)

            # Load V block: (BLOCK_N, HEAD_DIM)
            v_ptrs = (V_ptr
                      + batch_idx * v_stride_b
                      + kv_head_idx * v_stride_h
                      + k_offsets[:, None] * v_stride_s
                      + tl.arange(0, HEAD_DIM)[None, :])
            v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < K_SEQ_LEN, other=0.0).to(tl.float32)

            # Score: q_row (HEAD_DIM,) @ k_block (HEAD_DIM, BLOCK_N) → (BLOCK_N,)
            s_ij = tl.sum(q_row[:, None] * k_block, axis=0) * qk_scale

            # Mask: sink + causal + valid
            valid = (k_offsets < K_SEQ_LEN) & (k_offsets < SINK_SIZE) & (k_offsets <= Q_POS)
            s_ij = tl.where(valid, s_ij, -float('inf'))

            # Online softmax
            m_ij = tl.max(s_ij)
            m_new = tl.maximum(m_i, m_ij)
            scale_factor = tl.exp2(m_i - m_new)

            p_ij = tl.exp2(s_ij - m_new)
            p_ij = tl.where(valid, p_ij, 0.0)

            acc = acc * scale_factor + tl.sum(p_ij[:, None] * v_block, axis=0)
            l_i = l_i * scale_factor + tl.sum(p_ij)
            m_i = m_new

    # ================================================================
    # Phase 1: Window tokens (from window_start to causal_end)
    # ================================================================
    for start_n in range(window_start, causal_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K
        k_ptrs = (K_ptr
                  + batch_idx * k_stride_b
                  + kv_head_idx * k_stride_h
                  + k_offsets[None, :] * k_stride_s
                  + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < K_SEQ_LEN, other=0.0).to(tl.float32)

        # Load V
        v_ptrs = (V_ptr
                  + batch_idx * v_stride_b
                  + kv_head_idx * v_stride_h
                  + k_offsets[:, None] * v_stride_s
                  + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < K_SEQ_LEN, other=0.0).to(tl.float32)

        # Score
        s_ij = tl.sum(q_row[:, None] * k_block, axis=0) * qk_scale

        # Mask: causal + window + valid + not-a-sink (sinks handled above)
        dist = Q_POS - k_offsets
        valid = (k_offsets < K_SEQ_LEN) & (k_offsets <= Q_POS) & (dist < WINDOW_SIZE) & (dist >= 0)
        if SINK_SIZE > 0:
            valid = valid & (k_offsets >= SINK_SIZE)
        s_ij = tl.where(valid, s_ij, -float('inf'))

        # Online softmax
        row_has = tl.max(valid) > 0
        m_ij = tl.max(s_ij)
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

        p_ij = tl.where(valid, tl.exp2(s_ij - m_new), 0.0)

        acc = acc * scale_factor + tl.sum(p_ij[:, None] * v_block, axis=0)
        l_i = l_i * scale_factor + tl.sum(p_ij)
        m_i = m_new

    # ================================================================
    # 5. Normalise and store
    # ================================================================
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe

    o_ptrs = (O_ptr
              + batch_idx * q_stride_b
              + q_head_idx * q_stride_h
              + tl.arange(0, HEAD_DIM))
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty))
