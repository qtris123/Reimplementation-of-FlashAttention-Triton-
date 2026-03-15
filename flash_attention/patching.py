"""
Patching module for GPT-OSS models.

This module provides monkey-patching utilities to replace the default
eager_attention_forward in HuggingFace's GPT-OSS implementation with our
custom FlashAttention kernels (PyTorch or Triton).

Usage:
    from flash_attention.patching import patch_flash_attention, unpatch_flash_attention

    # Patch with Triton backend
    patch_flash_attention(backend="triton")

    # Now load and use GPT-OSS normally
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")

    # Unpatch when done
    unpatch_flash_attention()
"""

import torch
from typing import Optional, Tuple
from transformers.models.gpt_oss import modeling_gpt_oss

from .attention import FlashAttention
from .config import AttentionConfig


# Store original function for unpatching
_original_eager_attention_forward = modeling_gpt_oss.eager_attention_forward


def flash_attention_forward_pytorch(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Replacement for GPT-OSS eager_attention_forward using PyTorch backend.

    Args:
        module: The attention module from GPT-OSS
        query: (B, H_q, N_q, D)
        key: (B, H_kv, N_k, D)
        value: (B, H_kv, N_k, D)
        attention_mask: Not used (causal masking handled internally)
        scaling: Softmax scale (1/sqrt(d))
        dropout: Dropout rate (not supported in our kernels)

    Returns:
        (output, None) where output is (B, H_q, N_q, D)
    """
    # Get configuration from module
    sliding_window = getattr(module, 'sliding_window', None)

    # Determine number of KV heads from the key tensor
    num_kv_heads = key.shape[1]

    # Create attention config
    config = AttentionConfig(
        backend="pytorch",
        is_causal=True,
        num_kv_heads=num_kv_heads,
        window_size=sliding_window,
        sink_size=0,  # GPT-OSS doesn't use sinks by default
    )

    # Create FlashAttention module and run
    flash_attn = FlashAttention(config)
    output = flash_attn(query, key, value, q_pos_offset=0)

    return output, None


def flash_attention_forward_triton(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Replacement for GPT-OSS eager_attention_forward using Triton backend.

    Args:
        module: The attention module from GPT-OSS
        query: (B, H_q, N_q, D)
        key: (B, H_kv, N_k, D)
        value: (B, H_kv, N_k, D)
        attention_mask: Not used (causal masking handled internally)
        scaling: Softmax scale (1/sqrt(d))
        dropout: Dropout rate (not supported in our kernels)

    Returns:
        (output, None) where output is (B, H_q, N_q, D)
    """
    # Get configuration from module
    sliding_window = getattr(module, 'sliding_window', None)

    # Determine number of KV heads from the key tensor
    num_kv_heads = key.shape[1]

    # Create attention config
    config = AttentionConfig(
        backend="triton",
        is_causal=True,
        num_kv_heads=num_kv_heads,
        window_size=sliding_window,
        sink_size=0,  # GPT-OSS doesn't use sinks by default
    )

    # Create FlashAttention module and run
    flash_attn = FlashAttention(config)
    output = flash_attn(query, key, value, q_pos_offset=0)

    return output, None


def patch_flash_attention(backend: str = "triton"):
    """
    Patch HuggingFace's GPT-OSS attention implementation with our FlashAttention.

    Args:
        backend: "pytorch" or "triton" - which FlashAttention backend to use

    Example:
        >>> from flash_attention.patching import patch_flash_attention
        >>> patch_flash_attention(backend="triton")
        >>> model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
        >>> # model now uses our Triton FlashAttention kernels
    """
    if backend not in ("pytorch", "triton"):
        raise ValueError(f"backend must be 'pytorch' or 'triton', got '{backend}'")

    if backend == "triton":
        modeling_gpt_oss.eager_attention_forward = flash_attention_forward_triton
        print(f"✓ Patched GPT-OSS with FlashAttention (Triton backend)")
    else:
        modeling_gpt_oss.eager_attention_forward = flash_attention_forward_pytorch
        print(f"✓ Patched GPT-OSS with FlashAttention (PyTorch backend)")


def unpatch_flash_attention():
    """
    Restore the original GPT-OSS attention implementation.

    Example:
        >>> from flash_attention.patching import unpatch_flash_attention
        >>> unpatch_flash_attention()
        >>> # GPT-OSS models will now use the default HuggingFace implementation
    """
    modeling_gpt_oss.eager_attention_forward = _original_eager_attention_forward
    print("✓ Unpatched GPT-OSS (restored original attention)")
