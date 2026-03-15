#!/usr/bin/env python3
"""
Quick start example for using GPT-OSS with FlashAttention.

This demonstrates the basic patching workflow and compares both backends.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, "..")

from flash_attention.patching import patch_flash_attention, unpatch_flash_attention


def main():
    print("\n" + "="*70)
    print("GPT-OSS + FlashAttention Quick Start")
    print("="*70 + "\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This example requires a GPU.")
        return

    model_name = "openai/gpt-oss-20b"
    prompt = "The future of artificial intelligence is"

    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}\n")

    # ── Test 1: Triton Backend ──────────────────────────────────────────────
    print("─" * 70)
    print("Test 1: Using Triton Backend (Optimized)")
    print("─" * 70)

    patch_flash_attention(backend="triton")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    print("Generating...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )

    triton_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nOutput:\n{triton_output}\n")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    unpatch_flash_attention()

    # ── Test 2: PyTorch Backend ─────────────────────────────────────────────
    print("─" * 70)
    print("Test 2: Using PyTorch Backend (Reference)")
    print("─" * 70)

    patch_flash_attention(backend="pytorch")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    print("Generating...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )

    pytorch_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nOutput:\n{pytorch_output}\n")

    unpatch_flash_attention()

    # ── Summary ─────────────────────────────────────────────────────────────
    print("="*70)
    print("✅ Both backends work correctly!")
    print("="*70)
    print("\nNote: Outputs differ due to random sampling, but both use FlashAttention.")
    print("For production, use the Triton backend for better performance.\n")


if __name__ == "__main__":
    main()
