"""
KV Cache — pre-allocated buffer for autoregressive generation.

Stores K and V tensors for each layer, supporting incremental append
during decode steps.
"""

import torch


class KVCache:
    """
    Pre-allocated KV cache for autoregressive generation.

    Args:
        batch_size: Number of sequences in the batch
        n_kv_heads: Number of key/value heads
        max_seq_len: Maximum sequence length to pre-allocate
        head_dim: Dimension of each attention head
        dtype: Data type for the cache tensors
        device: Device for the cache tensors
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        self.max_seq_len = max_seq_len
        self.current_len = 0
        self.k_cache = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )

    def update(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V into the cache and return accumulated context.

        Args:
            k: (B, H_kv, S, D) — new key tokens
            v: (B, H_kv, S, D) — new value tokens
            start_pos: Position at which to write

        Returns:
            (k_full, v_full): (B, H_kv, start_pos + S, D) slices of the cache
        """
        seq_len = k.shape[2]
        end_pos = start_pos + seq_len

        assert end_pos <= self.max_seq_len, (
            f"Cache overflow: {end_pos} > {self.max_seq_len}"
        )

        self.k_cache[:, :, start_pos:end_pos, :] = k
        self.v_cache[:, :, start_pos:end_pos, :] = v
        self.current_len = end_pos

        return (
            self.k_cache[:, :, :end_pos, :],
            self.v_cache[:, :, :end_pos, :],
        )

    def reset(self):
        """Clear the cache for a new generation."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0
