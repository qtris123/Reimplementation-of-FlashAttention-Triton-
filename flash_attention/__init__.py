"""
flash_attention — Unified FlashAttention with switchable backends.

Public API:
    - FlashAttention   : nn.Module for attention computation
    - AttentionConfig  : dataclass for all configuration options
"""

from .config import AttentionConfig
from .attention import FlashAttention

__all__ = ["FlashAttention", "AttentionConfig"]
