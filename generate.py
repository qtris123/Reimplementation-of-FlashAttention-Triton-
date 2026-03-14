#!/usr/bin/env python3
"""
generate.py — Run Llama inference with our FlashAttention module.

Usage:
    python generate.py --prompt "Hello, how are you?"                        # default: 3B
    python generate.py --model meta-llama/Llama-3.2-1B --prompt "Hello"      # switch to 1B
    python generate.py --prompt "Hello" --backend pytorch                    # PyTorch backend
    python generate.py --prompt "Hello" --temperature 0                      # greedy decoding
"""

import argparse
import sys
import time
import torch

sys.path.insert(0, ".")
from flash_attention.model import LlamaModel, LlamaConfig
from flash_attention.weight_loader import load_hf_weights


# ── Model configs ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "meta-llama/Llama-3.2-1B":          dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8, head_dim=64),
    "meta-llama/Llama-3.2-1B-Instruct": dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8, head_dim=64),
    "meta-llama/Llama-3.2-3B":          dict(hidden_size=3072, intermediate_size=8192, num_hidden_layers=28, num_attention_heads=24, num_key_value_heads=8, head_dim=128),
    "meta-llama/Llama-3.2-3B-Instruct": dict(hidden_size=3072, intermediate_size=8192, num_hidden_layers=28, num_attention_heads=24, num_key_value_heads=8, head_dim=128),
}


def main():
    parser = argparse.ArgumentParser(description="Llama inference with FlashAttention")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--backend", type=str, default="triton", choices=["pytorch", "triton"],
                        help="Attention backend")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help=f"HuggingFace model ID. Supported: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"],
                        help="Model dtype")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("💥 CUDA not available. GPU is required.")
        sys.exit(1)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = "cuda"

    # ── Load tokenizer ──────────────────────────────────────────────────────
    print("📝 Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install transformers: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build and load model ────────────────────────────────────────────────
    if args.model not in MODEL_CONFIGS:
        print(f"⚠️  Unknown model '{args.model}', using default 1B config")
        model_kwargs = MODEL_CONFIGS["meta-llama/Llama-3.2-1B"]
    else:
        model_kwargs = MODEL_CONFIGS[args.model]

    print(f"🏗️  Building Llama model (backend={args.backend})...")
    config = LlamaConfig(**model_kwargs, attention_backend=args.backend)
    model = LlamaModel(config)

    model = load_hf_weights(model, model_name=args.model, device=device, dtype=dtype)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"   Parameters: {param_count:.1f}M | GPU memory: {mem_gb:.2f} GB")

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

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
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


if __name__ == "__main__":
    main()
