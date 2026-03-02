#!/usr/bin/env python3
"""
Test harness for the unified FlashAttention package.

Modeled after the original autograder.py. Tests every feature combination
across both backends (PyTorch and Triton) against a naive reference.

Usage:
    python test_flash_attention.py                       # run all tests
    python test_flash_attention.py --test noncausal      # specific test
    python test_flash_attention.py --test gqa_swa_sinks  # specific test
    python test_flash_attention.py --benchmark           # performance comparison
    python test_flash_attention.py --list                # list available tests
"""

import torch
import math
import argparse
import sys
import time
from dataclasses import dataclass

# ── Add parent dir to path so we can import the package ──────────────────────
sys.path.insert(0, ".")
from flash_attention import FlashAttention, AttentionConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Reference implementation (from autograder.py — source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    if num_groups == 1:
        return x
    B, H_kv, N, D = x.shape
    return x.unsqueeze(2).expand(B, H_kv, num_groups, N, D).reshape(B, H_kv * num_groups, N, D)


def _create_mask(seq_len, window_size, sink_size, device):
    idx = torch.arange(seq_len, device=device)
    row, col = idx.unsqueeze(1), idx.unsqueeze(0)
    sliding = (col <= row) & (col >= row - (window_size - 1))
    sink = (col < sink_size) & (col <= row) if sink_size > 0 else torch.zeros_like(sliding)
    return sliding | sink


def naive_attention(Q, K, V, is_causal=False, window_size=None, sink_size=None):
    """Ground-truth reference implementation."""
    B, H_q, N, D = Q.shape
    H_kv = K.shape[1]

    if H_q != H_kv:
        K = _repeat_kv(K, H_q // H_kv)
        V = _repeat_kv(V, H_q // H_kv)

    scale = 1.0 / math.sqrt(D)
    S = (Q @ K.transpose(-1, -2)) * scale

    if is_causal:
        w = window_size if window_size else N
        s = sink_size if sink_size else 0
        mask = _create_mask(N, w, s, Q.device)
        S.masked_fill_(~mask, -float('inf'))

    P = torch.nn.functional.softmax(S, dim=-1, dtype=torch.float32).to(Q.dtype)
    return P @ V


# ═══════════════════════════════════════════════════════════════════════════════
# Test definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    """A single test configuration."""
    name: str
    batch: int
    heads_q: int
    heads_kv: int
    seq_len: int
    head_dim: int
    is_causal: bool
    window_size: int | None = None
    sink_size: int = 0

    def to_config(self, backend: str) -> AttentionConfig:
        return AttentionConfig(
            backend=backend,
            is_causal=self.is_causal,
            num_kv_heads=self.heads_kv if self.heads_kv != self.heads_q else None,
            window_size=self.window_size,
            sink_size=self.sink_size,
        )

    @property
    def param_str(self):
        parts = [f"B={self.batch}", f"Hq={self.heads_q}"]
        if self.heads_kv != self.heads_q:
            parts.append(f"Hkv={self.heads_kv}")
        parts += [f"N={self.seq_len}", f"D={self.head_dim}"]
        if self.window_size:
            parts.append(f"W={self.window_size}")
        if self.sink_size:
            parts.append(f"S={self.sink_size}")
        return f"({', '.join(parts)})"


# ── Test groups ──────────────────────────────────────────────────────────────

TEST_GROUPS = {
    "noncausal": {
        "label": "Non-Causal MHA",
        "cases": [
            TestCase("nc_256",  1, 4,  4,  256,  16, False),
            TestCase("nc_512",  1, 8,  8,  512,  16, False),
        ],
    },
    "causal": {
        "label": "Causal MHA",
        "cases": [
            TestCase("c_256",  1, 4,  4,  256,  16, True),
            TestCase("c_512",  1, 8,  8,  512,  16, True),
        ],
    },
    "gqa": {
        "label": "Grouped-Query Attention (Causal)",
        "cases": [
            TestCase("gqa_256",  1, 8,  2, 256,  16, True),
            TestCase("gqa_512",  1, 8,  2, 512,  16, True),
        ],
    },
    "swa": {
        "label": "Sliding Window + GQA (Causal)",
        "cases": [
            TestCase("swa_256",  1, 8,  2, 256,  16, True, window_size=128),
            TestCase("swa_512",  1, 8,  2, 512,  16, True, window_size=128),
        ],
    },
    "gqa_swa_sinks": {
        "label": "GQA + Sliding Window + Attention Sinks (Causal)",
        "cases": [
            TestCase("sink_256",  1, 8,  2, 256,  32, True, window_size=128, sink_size=8),
            TestCase("sink_512",  1, 8,  2, 512,  16, True, window_size=128, sink_size=8),
        ],
    },
    "backend_switch": {
        "label": "Backend Switching Consistency",
        "cases": [
            TestCase("switch_causal",  1, 4,  4, 256, 16, True),
            TestCase("switch_gqa",     1, 8,  2, 256, 16, True),
            TestCase("switch_swa",     1, 8,  2, 256, 16, True, window_size=128),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Test runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_test(tc: TestCase, backend: str) -> bool:
    """Run one test case on one backend. Returns True if passed."""
    torch.manual_seed(42)
    q = torch.randn(tc.batch, tc.heads_q, tc.seq_len, tc.head_dim, device='cuda', dtype=DTYPE)
    k = torch.randn(tc.batch, tc.heads_kv, tc.seq_len, tc.head_dim, device='cuda', dtype=DTYPE)
    v = torch.randn(tc.batch, tc.heads_kv, tc.seq_len, tc.head_dim, device='cuda', dtype=DTYPE)

    # Reference
    ref = naive_attention(q, k, v, is_causal=tc.is_causal,
                          window_size=tc.window_size, sink_size=tc.sink_size if tc.sink_size else None)

    # Student
    config = tc.to_config(backend)
    attn = FlashAttention(config)
    out = attn(q, k, v)

    passed = torch.allclose(ref, out, rtol=5e-2, atol=5e-2)
    status = "✅" if passed else "❌"
    print(f"  {status} [{backend:>7s}] {tc.param_str}")

    if not passed:
        max_diff = (ref - out).abs().max().item()
        print(f"         Max diff: {max_diff:.6f}")

    return passed


def run_backend_switch_test(tc: TestCase) -> bool:
    """Verify both backends produce the same output."""
    torch.manual_seed(42)
    q = torch.randn(tc.batch, tc.heads_q, tc.seq_len, tc.head_dim, device='cuda', dtype=DTYPE)
    k = torch.randn(tc.batch, tc.heads_kv, tc.seq_len, tc.head_dim, device='cuda', dtype=DTYPE)
    v = torch.randn(tc.batch, tc.heads_kv, tc.seq_len, tc.head_dim, device='cuda', dtype=DTYPE)

    config = tc.to_config("triton")
    attn = FlashAttention(config)

    out_triton = attn(q, k, v)
    attn.switch_backend("pytorch")
    out_pytorch = attn(q, k, v)

    # Compare both against reference
    ref = naive_attention(q, k, v, is_causal=tc.is_causal,
                          window_size=tc.window_size, sink_size=tc.sink_size if tc.sink_size else None)

    triton_ok = torch.allclose(ref, out_triton, rtol=5e-2, atol=5e-2)
    pytorch_ok = torch.allclose(ref, out_pytorch, rtol=5e-2, atol=5e-2)
    consistent = torch.allclose(out_triton, out_pytorch, rtol=5e-2, atol=5e-2)

    passed = triton_ok and pytorch_ok and consistent
    status = "✅" if passed else "❌"
    print(f"  {status} [switch ] {tc.param_str}  triton={triton_ok}, pytorch={pytorch_ok}, match={consistent}")

    if not passed:
        if not triton_ok:
            print(f"         Triton max diff:  {(ref - out_triton).abs().max():.6f}")
        if not pytorch_ok:
            print(f"         PyTorch max diff: {(ref - out_pytorch).abs().max():.6f}")

    return passed


def run_test_group(group_name: str) -> bool:
    """Run all test cases in a group."""
    group = TEST_GROUPS[group_name]
    print(f"\n{'='*60}")
    print(f"  {group['label']}")
    print(f"{'='*60}")

    all_passed = True

    if group_name == "backend_switch":
        for tc in group["cases"]:
            if not run_backend_switch_test(tc):
                all_passed = False
    else:
        for tc in group["cases"]:
            for backend in ("triton", "pytorch"):
                if not run_single_test(tc, backend):
                    all_passed = False

    if all_passed:
        print(f"\n  🎉 All {group['label']} tests passed!")
    else:
        print(f"\n  ⚠️  Some {group['label']} tests failed.")

    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    """Compare PyTorch vs Triton backend performance."""
    print(f"\n{'='*70}")
    print("  Performance Benchmark: PyTorch vs Triton")
    print(f"{'='*70}")

    configs = [
        ("Causal MHA",      AttentionConfig(is_causal=True),                                     8, 8,  4096, 16),
        ("Causal GQA",      AttentionConfig(is_causal=True, num_kv_heads=2),                     16, 2, 4096, 16),
        ("GQA + SWA",       AttentionConfig(is_causal=True, num_kv_heads=2, window_size=128),    16, 2, 4096, 16),
        ("GQA+SWA+Sinks",   AttentionConfig(is_causal=True, num_kv_heads=2, window_size=128, sink_size=8), 16, 2, 4096, 16),
    ]

    print(f"\n{'Config':<20} | {'Backend':<10} | {'Avg (ms)':<12} | {'Peak Mem (GB)':<15}")
    print("-" * 65)

    for label, config, hq, hkv, seq_len, dim in configs:
        q = torch.randn(1, hq, seq_len, dim, device='cuda', dtype=DTYPE)
        k = torch.randn(1, hkv, seq_len, dim, device='cuda', dtype=DTYPE)
        v = torch.randn(1, hkv, seq_len, dim, device='cuda', dtype=DTYPE)

        for backend in ("triton", "pytorch"):
            config.backend = backend
            attn = FlashAttention(config)

            # Warm up
            for _ in range(3):
                attn(q, k, v)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            for _ in range(20):
                attn(q, k, v)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000 / 20
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

            print(f"{label:<20} | {backend:<10} | {elapsed:<12.4f} | {peak_mem:<15.4f}")

        print("-" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test harness for the unified FlashAttention package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", type=str, default=None,
                        help="Run a specific test group (e.g. causal, gqa, swa, gqa_swa_sinks, backend_switch)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    parser.add_argument("--list", action="store_true",
                        help="List available test groups")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("💥 CUDA not available. Skipping all GPU tests.")
        sys.exit(1)

    if args.list:
        print("\nAvailable test groups:")
        for name, group in TEST_GROUPS.items():
            print(f"  {name:<20s} — {group['label']} ({len(group['cases'])} cases)")
        return

    if args.benchmark:
        run_benchmark()
        return

    if args.test:
        if args.test not in TEST_GROUPS:
            print(f"Unknown test group: '{args.test}'. Use --list to see options.")
            sys.exit(1)
        run_test_group(args.test)
    else:
        # Run all tests
        all_ok = True
        for name in TEST_GROUPS:
            if not run_test_group(name):
                all_ok = False

        if all_ok:
            print(f"\n{'='*60}")
            print("  🏆 ALL TESTS PASSED!")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("  ⚠️  SOME TESTS FAILED — see above for details.")
            print(f"{'='*60}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
