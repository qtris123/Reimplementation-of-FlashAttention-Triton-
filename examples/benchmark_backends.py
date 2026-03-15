#!/usr/bin/env python3
"""
Benchmark PyTorch vs Triton backends on attention computation.

This script measures throughput and memory usage for both backends
across different sequence lengths and batch sizes.
"""

import torch
import time
from dataclasses import dataclass
import sys
sys.path.insert(0, "..")

from flash_attention.attention import FlashAttention
from flash_attention.config import AttentionConfig


@dataclass
class BenchmarkResult:
    backend: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    time_ms: float
    memory_mb: float

    def throughput_tokens_per_sec(self):
        total_tokens = self.batch_size * self.seq_len
        return total_tokens / (self.time_ms / 1000)


def benchmark_attention(
    backend: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_warmup: int = 10,
    num_iters: int = 100,
):
    """Benchmark a single attention configuration."""

    # Create config
    config = AttentionConfig(
        backend=backend,
        is_causal=True,
        num_kv_heads=num_heads,
    )

    # Create tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

    # Create attention module
    attn = FlashAttention(config)

    # Warmup
    for _ in range(num_warmup):
        _ = attn(Q, K, V)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = attn(Q, K, V)
    torch.cuda.synchronize()
    end = time.perf_counter()

    # Compute metrics
    time_ms = (end - start) * 1000 / num_iters
    memory_mb = torch.cuda.max_memory_allocated() / 1e6

    return BenchmarkResult(
        backend=backend,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        time_ms=time_ms,
        memory_mb=memory_mb,
    )


def main():
    print("\n" + "="*80)
    print("FlashAttention Backend Benchmark")
    print("="*80 + "\n")

    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires a GPU.")
        return

    # Benchmark configurations
    configs = [
        (1, 512, 32, 64),    # Small sequence
        (1, 1024, 32, 64),   # Medium sequence
        (1, 2048, 32, 64),   # Large sequence
        (1, 4096, 32, 64),   # Very large sequence
        (2, 1024, 32, 64),   # Batch size 2
        (4, 1024, 32, 64),   # Batch size 4
    ]

    results = []

    for batch_size, seq_len, num_heads, head_dim in configs:
        print(f"\nBenchmarking B={batch_size}, N={seq_len}, H={num_heads}, D={head_dim}")
        print("-" * 80)

        # Benchmark PyTorch
        try:
            result_pytorch = benchmark_attention("pytorch", batch_size, seq_len, num_heads, head_dim)
            results.append(result_pytorch)
            print(f"PyTorch: {result_pytorch.time_ms:.2f}ms | {result_pytorch.memory_mb:.1f}MB | "
                  f"{result_pytorch.throughput_tokens_per_sec():.0f} tok/s")
        except Exception as e:
            print(f"PyTorch: Failed ({e})")

        # Benchmark Triton
        try:
            result_triton = benchmark_attention("triton", batch_size, seq_len, num_heads, head_dim)
            results.append(result_triton)
            print(f"Triton:  {result_triton.time_ms:.2f}ms | {result_triton.memory_mb:.1f}MB | "
                  f"{result_triton.throughput_tokens_per_sec():.0f} tok/s")

            # Speedup
            if result_pytorch:
                speedup = result_pytorch.time_ms / result_triton.time_ms
                print(f"Speedup: {speedup:.2f}x (Triton vs PyTorch)")
        except Exception as e:
            print(f"Triton: Failed ({e})")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"{'Backend':<10} {'B':<5} {'N':<6} {'H':<5} {'D':<5} {'Time (ms)':<12} {'Memory (MB)':<15} {'Tok/s':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r.backend:<10} {r.batch_size:<5} {r.seq_len:<6} {r.num_heads:<5} {r.head_dim:<5} "
              f"{r.time_ms:<12.2f} {r.memory_mb:<15.1f} {r.throughput_tokens_per_sec():<10.0f}")

    print("\n")


if __name__ == "__main__":
    main()
