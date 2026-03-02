"""
Triton kernel — causal FlashAttention with GQA + Sliding Window + Attention Sinks.

Synthesised from problems 4, 5, 6, and 7.

Three-phase processing:
  Phase 0  — Attention sink tokens         (if SINK_SIZE > 0)
  Phase 1  — Off-diagonal window blocks    (past tokens within window, excluding sinks)
  Phase 2  — Diagonal blocks               (causal + window masking)

When WINDOW_SIZE == SEQ_LEN and SINK_SIZE == 0, this degenerates to standard
causal attention (problem 4/5 behaviour). Triton's constexpr folding optimises
away the dead branches.
"""

import triton
import triton.language as tl


@triton.jit
def flash_causal_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Scalars
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Compile-time constants
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE:   tl.constexpr,
    HEAD_DIM:    tl.constexpr,
    BLOCK_M:     tl.constexpr,
    BLOCK_N:     tl.constexpr,
):
    """
    General causal flash attention kernel.

    Feature matrix controlled by constexprs:
      - GQA:     N_Q_HEADS != N_KV_HEADS  (head remapping)
      - SWA:     WINDOW_SIZE < SEQ_LEN    (sliding window)
      - Sinks:   SINK_SIZE > 0            (attention sinks)
    """
    # ================================================================
    # 1. Block / head identification
    # ================================================================
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)

    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # GQA head mapping
    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # ================================================================
    # 2. Initialise accumulators in SRAM
    # ================================================================
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ================================================================
    # 3. Load query block
    # ================================================================
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = (Q_ptr
              + batch_idx * q_stride_b
              + q_head_idx * q_stride_h
              + q_offsets[:, None] * q_stride_s
              + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    q_block = q_block.to(tl.float32)

    qk_scale = softmax_scale * 1.44269504
    diag_start = q_block_idx * BLOCK_M

    # ================================================================
    # Phase 0: Attention sink tokens (first SINK_SIZE positions)
    # ================================================================
    if SINK_SIZE > 0:
        for start_n in range(0, SINK_SIZE, BLOCK_N):
            k_offsets = start_n + tl.arange(0, BLOCK_N)

            # Load K
            k_ptrs = (K_ptr
                      + batch_idx * k_stride_b
                      + kv_head_idx * k_stride_h
                      + k_offsets[None, :] * k_stride_s
                      + tl.arange(0, HEAD_DIM)[:, None])
            k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0).to(tl.float32)

            # Load V
            v_ptrs = (V_ptr
                      + batch_idx * v_stride_b
                      + kv_head_idx * v_stride_h
                      + k_offsets[:, None] * v_stride_s
                      + tl.arange(0, HEAD_DIM)[None, :])
            v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0).to(tl.float32)

            # Scores
            s_ij = tl.dot(q_block, k_block) * qk_scale

            # Mask: sink columns, causal, valid
            sink_cols = k_offsets[None, :] < SINK_SIZE
            causal = q_offsets[:, None] >= k_offsets[None, :]
            valid = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
            mask = sink_cols & causal & valid
            s_ij = tl.where(mask, s_ij, -float('inf'))

            # Online softmax (skip rows with nothing valid)
            row_has = tl.max(mask, axis=1) > 0
            m_ij = tl.max(s_ij, axis=1)
            m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
            scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

            p_ij = tl.where(row_has[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)
            acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
            l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
            m_i = m_new

    # ================================================================
    # Phase 1: Off-diagonal blocks (within window, excluding sinks)
    # ================================================================
    # Window start: max(0, diag_start - WINDOW_SIZE + 1)
    win_left = diag_start - (WINDOW_SIZE - 1)
    window_start = tl.maximum(0, win_left)

    for start_n in range(window_start, diag_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K
        k_ptrs = (K_ptr
                  + batch_idx * k_stride_b
                  + kv_head_idx * k_stride_h
                  + k_offsets[None, :] * k_stride_s
                  + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0).to(tl.float32)

        # Load V
        v_ptrs = (V_ptr
                  + batch_idx * v_stride_b
                  + kv_head_idx * v_stride_h
                  + k_offsets[:, None] * v_stride_s
                  + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0).to(tl.float32)

        # Scores
        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Combined mask: window + validity + pre-diagonal + exclude sinks
        dist = q_offsets[:, None] - k_offsets[None, :]
        window_mask = (dist >= 0) & (dist < WINDOW_SIZE)
        valid_mask = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        pre_diag = k_offsets[None, :] < diag_start

        if SINK_SIZE > 0:
            non_sink = k_offsets[None, :] >= SINK_SIZE
            mask = window_mask & valid_mask & pre_diag & non_sink
        else:
            mask = window_mask & valid_mask & pre_diag

        s_ij = tl.where(mask, s_ij, -float('inf'))

        # Online softmax
        row_has = tl.max(mask, axis=1) > 0
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

        p_ij = tl.where(row_has[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)
        acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
        l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
        m_i = m_new

    # ================================================================
    # Phase 2: Diagonal blocks (with causal + window masking)
    # ================================================================
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K
        k_ptrs = (K_ptr
                  + batch_idx * k_stride_b
                  + kv_head_idx * k_stride_h
                  + k_offsets[None, :] * k_stride_s
                  + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0).to(tl.float32)

        # Load V
        v_ptrs = (V_ptr
                  + batch_idx * v_stride_b
                  + kv_head_idx * v_stride_h
                  + k_offsets[:, None] * v_stride_s
                  + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0).to(tl.float32)

        # Scores
        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Mask: causal + window + validity (+ exclude already-handled sinks)
        causal = q_offsets[:, None] >= k_offsets[None, :]
        valid = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        dist = q_offsets[:, None] - k_offsets[None, :]
        window_mask = (dist >= 0) & (dist < WINDOW_SIZE)

        if SINK_SIZE > 0:
            non_sink = k_offsets[None, :] >= SINK_SIZE
            mask = causal & valid & window_mask & non_sink
        else:
            mask = causal & valid & window_mask

        s_ij = tl.where(mask, s_ij, -float('inf'))

        # Online softmax
        row_has = tl.max(mask, axis=1) > 0
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

        p_ij = tl.where(row_has[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)
        acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
        l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
        m_i = m_new

    # ================================================================
    # 4. Normalise and write output
    # ================================================================
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_ptrs = (O_ptr
              + batch_idx * q_stride_b
              + q_head_idx * q_stride_h
              + q_offsets[:, None] * q_stride_s
              + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)
