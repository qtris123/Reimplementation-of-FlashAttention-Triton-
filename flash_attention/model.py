"""
Llama model architecture — supports Llama-3.2-1B and Llama-3.2-3B.

Uses our FlashAttention module for the attention computation.
All other components (projections, MLP, norms) are standard PyTorch.

Supported models:
  - Llama-3.2-1B:  16 layers, 32 Q heads, 8 KV heads, hidden=2048, head_dim=64
  - Llama-3.2-3B:  28 layers, 24 Q heads, 8 KV heads, hidden=3072, head_dim=128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .config import AttentionConfig
from .attention import FlashAttention
from .kv_cache import KVCache
from .rope import precompute_freqs_cis, apply_rotary_emb


# ═══════════════════════════════════════════════════════════════════════════════
# Model Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LlamaConfig:
    """Configuration for the Llama model."""
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32      # Q heads
    num_key_value_heads: int = 8       # KV heads (GQA)
    head_dim: int = 64
    vocab_size: int = 128256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    attention_backend: str = "triton"  # "pytorch" or "triton"


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════════════════════════

class LlamaAttention(nn.Module):
    """
    Llama attention layer with QKV projections, RoPE, KV cache,
    and our FlashAttention module.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.n_q_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_q_heads * self.head_dim, self.hidden_size, bias=False)

        # Flash attention module
        attn_config = AttentionConfig(
            backend=config.attention_backend,
            is_causal=True,
            num_kv_heads=self.n_kv_heads,
        )
        self.flash_attn = FlashAttention(attn_config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) — hidden states
            freqs_cis: (S, head_dim//2) — RoPE frequencies for these positions
            start_pos: Starting position in the sequence (for KV cache)
            kv_cache: Optional KV cache for incremental decoding

        Returns:
            (B, S, D) — output hidden states
        """
        B, S, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, S, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, H_q, S, D), k/v: (B, H_kv, S, D)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # KV cache: append and get full context
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)
            # k, v now span [0, start_pos + S)

        # Attention — FlashAttention handles GQA internally
        attn_out = self.flash_attn(q, k, v, q_pos_offset=start_pos)
        # attn_out: (B, H_q, S, D)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)


# ═══════════════════════════════════════════════════════════════════════════════
# MLP (SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════════

class LlamaMLP(nn.Module):
    """Llama MLP with SiLU gating (SwiGLU)."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════════════

class LlamaBlock(nn.Module):
    """Single Llama transformer block: attention + MLP with residuals."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.attention(self.input_layernorm(x), freqs_cis, start_pos, kv_cache)
        # Pre-norm MLP with residual
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class LlamaModel(nn.Module):
    """
    Complete Llama-3.2-1B model.

    Components:
      - Token embedding
      - N transformer blocks (attention + MLP)
      - Final RMSNorm
      - LM head (output projection)
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            LlamaBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # LM head (may share weights with embed_tokens)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis: torch.Tensor | None = None

        # KV caches (one per layer)
        self.kv_caches: list[KVCache] = []

    def _ensure_freqs(self, device: torch.device):
        """Lazily compute RoPE frequencies on the right device."""
        if self.freqs_cis is None or self.freqs_cis.device != device:
            self.freqs_cis = precompute_freqs_cis(
                self.config.head_dim,
                self.config.max_position_embeddings,
                self.config.rope_theta,
                device=device,
            )

    def setup_caches(self, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        """Initialize KV caches for all layers."""
        self.kv_caches = [
            KVCache(
                batch_size,
                self.config.num_key_value_heads,
                max_seq_len,
                self.config.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.config.num_hidden_layers)
        ]

    def clear_caches(self):
        """Reset all KV caches."""
        for cache in self.kv_caches:
            cache.reset()

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            token_ids: (B, S) — input token IDs
            start_pos: Starting position for RoPE + KV cache

        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = token_ids.shape
        device = token_ids.device
        self._ensure_freqs(device)

        # Embedding
        h = self.embed_tokens(token_ids)  # (B, S, D)

        # Slice RoPE frequencies for current positions
        freqs_cis = self.freqs_cis[start_pos : start_pos + S]

        # Transformer blocks
        use_cache = len(self.kv_caches) > 0
        for i, layer in enumerate(self.layers):
            kv_cache = self.kv_caches[i] if use_cache else None
            h = layer(h, freqs_cis, start_pos, kv_cache)

        # Final norm + LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt_ids: (B, prompt_len) — tokenised prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            eos_token_id: Stop token ID

        Returns:
            (B, prompt_len + generated_len) — full sequence of token IDs
        """
        B, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        dtype = next(self.parameters()).dtype

        max_seq = prompt_len + max_new_tokens
        self.setup_caches(B, max_seq, device, dtype)

        # Prefill: process entire prompt at once
        logits = self(prompt_ids, start_pos=0)
        # Only need logits for the last token
        next_logits = logits[:, -1, :]

        generated = [prompt_ids]

        for step in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    probs = _top_p_filter(probs, top_p)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            generated.append(next_token)

            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Decode: single token forward
            cur_pos = prompt_len + step
            next_logits = self(next_token, start_pos=cur_pos)[:, -1, :]

        self.clear_caches()
        return torch.cat(generated, dim=1)


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to probabilities."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens with cumulative probability above top_p
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    # Re-normalise
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    # Scatter back
    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
    return probs
