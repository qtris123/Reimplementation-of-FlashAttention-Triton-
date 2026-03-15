#!/usr/bin/env python3
"""
generate_gpt_oss.py — Run GPT-OSS inference with FlashAttention.

This script demonstrates how to use our custom FlashAttention kernels
(PyTorch or Triton backend) with OpenAI's GPT-OSS models via monkey-patching.

Usage:
    python generate_gpt_oss.py --prompt "Hello, how are you?"
    python generate_gpt_oss.py --model openai/gpt-oss-20b --prompt "Explain quantum computing"
    python generate_gpt_oss.py --prompt "Hello" --backend pytorch  # Use PyTorch backend
    python generate_gpt_oss.py --prompt "Hello" --temperature 0     # Greedy decoding
"""

import argparse
import sys
import time
import torch

sys.path.insert(0, ".")
from flash_attention.patching import patch_flash_attention, unpatch_flash_attention


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS inference with FlashAttention")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--backend", type=str, default="triton", choices=["pytorch", "triton"],
                        help="FlashAttention backend to use")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="HuggingFace model ID (e.g., openai/gpt-oss-20b, openai/gpt-oss-120b)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"],
                        help="Model dtype")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("💥 CUDA not available. GPU is required.")
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = "cuda"

    # ── Patch FlashAttention ────────────────────────────────────────────────
    print(f"🔧 Patching GPT-OSS with FlashAttention ({args.backend} backend)...")
    patch_flash_attention(backend=args.backend)

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print("📝 Loading tokenizer...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
    except ImportError:
        print("Install transformers: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"🏗️  Loading GPT-OSS model: {args.model}")
    print(f"   Backend: {args.backend} | dtype: {args.dtype}")

    # Configure MXFP4 quantization (dequantize=True loads weights then dequantizes to dtype)
    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # Required for patching to work
        quantization_config=quantization_config,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Parameters: {param_count:.1f}B | GPU memory: {mem_gb:.2f} GB")

    # ── Tokenize prompt ─────────────────────────────────────────────────────
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]
    print(f"\n📨 Prompt ({prompt_len} tokens): {args.prompt}")
    print(f"🔧 Settings: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print(f"\n{'─'*60}")
    print("🤖 Generating...\n")

    # ── Generate ────────────────────────────────────────────────────────────
    start_time = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = time.time() - start_time
    new_tokens = output_ids.shape[1] - prompt_len

    # ── Decode and print ────────────────────────────────────────────────────
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

    print(generated_text)
    print(f"\n{'─'*60}")
    print(f"📊 Stats: {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
    print(f"   Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    unpatch_flash_attention()


if __name__ == "__main__":
    main()
