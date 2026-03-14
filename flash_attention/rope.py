"""
Rotary Position Embeddings (RoPE) for Llama-style models.

Precomputes frequency tensors and applies rotary embeddings
to query and key tensors before attention.
"""

import torch
import math


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Precompute the complex exponential frequencies for RoPE.

    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        theta: Base frequency (Llama 3.x uses 500000.0)
        device: Device

    Returns:
        freqs_cis: (max_seq_len, head_dim // 2) complex64
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to a tensor.

    Args:
        x: (B, H, S, D) — queries or keys
        freqs_cis: (S, D // 2) complex64 — pre-sliced to the right positions

    Returns:
        (B, H, S, D) with rotary embeddings applied
    """
    # Reshape x to pairs: (B, H, S, D//2, 2)
    x_float = x.float()
    x_pairs = x_float.reshape(*x_float.shape[:-1], -1, 2)
    # Convert to complex
    x_complex = torch.view_as_complex(x_pairs)  # (B, H, S, D//2)
    # Apply rotation: broadcast freqs_cis (S, D//2) → (1, 1, S, D//2)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D//2)
    x_rotated = x_complex * freqs
    # Back to real
    x_out = torch.view_as_real(x_rotated).reshape_as(x_float)
    return x_out.to(x.dtype)
