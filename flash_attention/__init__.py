"""
flash_attention — Unified FlashAttention with switchable backends.

Public API:
    - FlashAttention   : nn.Module for attention computation
    - AttentionConfig  : dataclass for all configuration options
    - KVCache          : pre-allocated KV cache for generation
    - LlamaModel       : complete Llama model for inference
    - LlamaConfig      : Llama model configuration
"""

from .config import AttentionConfig
from .attention import FlashAttention
from .kv_cache import KVCache
from .model import LlamaModel, LlamaConfig

__all__ = [
    "FlashAttention",
    "AttentionConfig",
    "KVCache",
    "LlamaModel",
    "LlamaConfig",
]
