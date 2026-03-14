"""
Shared utilities used by both backends.
"""

import math
import torch


def compute_softmax_scale(head_dim: int) -> float:
    """Standard attention scaling factor: 1 / sqrt(d_k)."""
    return 1.0 / math.sqrt(head_dim)


def repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Repeat K/V heads to match the number of query heads for GQA.

    Args:
        x: (B, H_kv, N, D)
        num_groups: number of query heads per KV head

    Returns:
        (B, H_kv * num_groups, N, D)
    """
    if num_groups == 1:
        return x
    B, H_kv, N, D = x.shape
    x = x.unsqueeze(2).expand(B, H_kv, num_groups, N, D)
    return x.reshape(B, H_kv * num_groups, N, D)


def create_attention_mask(
    seq_len: int,
    is_causal: bool = True,
    window_size: int | None = None,
    sink_size: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a boolean attention mask.

    Combines causal, sliding-window, and attention-sink constraints
    into a single (seq_len, seq_len) boolean tensor.

    Returns:
        (seq_len, seq_len) bool tensor — True where attention is allowed.
    """
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)  # (seq_len, 1) — query positions
    col = idx.unsqueeze(0)  # (1, seq_len) — key positions

    if not is_causal:
        if window_size is None:
            return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        raise ValueError("Sliding window requires causal masking")

    # Effective window — full sequence when not using sliding window
    w = window_size if window_size is not None else seq_len

    # Sliding window: i - (w-1) <= j <= i
    sliding = (col <= row) & (col >= row - (w - 1))

    # Sink tokens: j < sink_size AND j <= i
    if sink_size > 0:
        sink = (col < sink_size) & (col <= row)
        return sliding | sink

    return sliding


def validate_qkv_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_kv_heads: int | None = None):
    """Validate Q/K/V tensor shapes and raise clear errors."""
    if q.dim() != 4:
        raise ValueError(f"Expected 4-D tensors (B, H, N, D), got Q with {q.dim()} dims")
    B, H_q, N_q, D = q.shape
    _, H_kv, N_k, D_k = k.shape

    if k.shape != v.shape:
        raise ValueError(f"K and V shapes must match: K={k.shape}, V={v.shape}")
    if N_q > N_k:
        raise ValueError(f"Q seq_len must be <= K seq_len: Q_seq={N_q}, K_seq={N_k}")
    if D != D_k:
        raise ValueError(f"Q and K head dims must match: Q_dim={D}, K_dim={D_k}")
    if B != k.shape[0]:
        raise ValueError(f"Batch sizes must match: Q_batch={B}, K_batch={k.shape[0]}")

    if num_kv_heads is not None:
        if H_kv != num_kv_heads:
            raise ValueError(
                f"K has {H_kv} heads but config.num_kv_heads={num_kv_heads}"
            )
        if H_q % H_kv != 0:
            raise ValueError(
                f"num_q_heads ({H_q}) must be divisible by num_kv_heads ({H_kv})"
            )
    elif H_q != H_kv:
        raise ValueError(
            f"Q has {H_q} heads, K has {H_kv} heads. "
            f"Set config.num_kv_heads for GQA."
        )

