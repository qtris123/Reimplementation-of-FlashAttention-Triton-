"""
PyTorch backend — pure-Python tiled FlashAttention (reference implementation).

Adapted from problem_1.py. Supports all feature combinations:
  - Causal / non-causal
  - MHA / GQA
  - Sliding window attention
  - Attention sinks
  - Prefill (Q_seq == K_seq) and Decode (Q_seq <= K_seq)

This backend is useful for debugging and as a correctness reference.
It is NOT optimised for speed — use the Triton backend for production.
"""

import math
import torch

from ..config import AttentionConfig
from ..utils import compute_softmax_scale, create_attention_mask


def pytorch_flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: AttentionConfig,
    q_pos_offset: int = 0,
) -> torch.Tensor:
    """
    Tiled FlashAttention-2 forward pass in pure PyTorch.

    Args:
        Q: (B, H_q, N_q, D)
        K: (B, H_kv, N_k, D)
        V: (B, H_kv, N_k, D)
        config: AttentionConfig controlling behaviour
        q_pos_offset: Position offset for Q tokens in the full sequence.
                      During decode, Q may be 1 token at position `start_pos`.

    Returns:
        O: (B, H_q, N_q, D) — attention output
    """
    B, H_q, N_q, D = Q.shape
    H_kv = K.shape[1]
    N_k = K.shape[2]
    num_groups = H_q // H_kv  # 1 for MHA, >1 for GQA

    Q_TILE = config.block_m
    K_TILE = config.block_n
    scale = compute_softmax_scale(D)

    N_Q_tiles = math.ceil(N_q / Q_TILE)
    N_K_tiles = math.ceil(N_k / K_TILE)

    O_final = torch.zeros_like(Q, dtype=Q.dtype)

    for b in range(B):
        for h_q in range(H_q):
            # GQA: map query head to its shared KV head
            h_kv = h_q // num_groups

            Q_bh = Q[b, h_q, :, :]
            K_bh = K[b, h_kv, :, :]
            V_bh = V[b, h_kv, :, :]

            for i in range(N_Q_tiles):
                q_start = i * Q_TILE
                q_end = min((i + 1) * Q_TILE, N_q)
                Q_tile = Q_bh[q_start:q_end, :]

                # Running accumulators for online softmax
                o_i = torch.zeros_like(Q_tile, dtype=Q.dtype)
                l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
                m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32)

                for j in range(N_K_tiles):
                    k_start = j * K_TILE
                    k_end = min((j + 1) * K_TILE, N_k)

                    K_tile = K_bh[k_start:k_end, :]
                    V_tile = V_bh[k_start:k_end, :]

                    # Attention scores
                    S_ij = (Q_tile @ K_tile.transpose(-1, -2)) * scale

                    # ---- Masking ----
                    if config.is_causal:
                        # q_idx: absolute positions in the full sequence
                        q_idx = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1) + q_pos_offset
                        k_idx = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)

                        # Causal: allow k <= q
                        mask = k_idx <= q_idx

                        # Sliding window: only attend within window
                        if config.use_sliding_window:
                            dist = q_idx - k_idx
                            window_mask = dist < config.window_size
                            mask = mask & window_mask

                            # Attention sinks: always attend to first sink_size tokens
                            if config.use_sinks:
                                sink_mask = k_idx < config.sink_size
                                causal_sink = sink_mask & (k_idx <= q_idx)
                                mask = mask | causal_sink

                        S_ij = S_ij.masked_fill(~mask, -float('inf'))

                    # ---- Online softmax update ----
                    m_ij = torch.max(S_ij, dim=-1).values.to(torch.float32)
                    m_new = torch.maximum(m_i, m_ij)

                    # Guard: when m_i and m_new are both -inf, exp(-inf - -inf)=NaN → use 0
                    scale_factor = torch.nan_to_num(torch.exp(m_i - m_new), nan=0.0)
                    o_i = o_i * scale_factor.unsqueeze(-1).to(o_i.dtype)
                    l_i = l_i * scale_factor

                    P_tilde = torch.nan_to_num(
                        torch.exp(S_ij.to(torch.float32) - m_new.unsqueeze(-1)),
                        nan=0.0,
                    )
                    l_i = l_i + torch.sum(P_tilde, dim=-1)
                    o_i = o_i + (P_tilde @ V_tile.to(torch.float32)).to(o_i.dtype)

                    m_i = m_new

                # Normalise after all KV tiles
                l_i_safe = torch.where(l_i == 0, torch.ones_like(l_i), l_i)
                o_i = o_i / l_i_safe.unsqueeze(-1)
                O_final[b, h_q, q_start:q_end, :] = o_i

    return O_final.to(Q.dtype)

