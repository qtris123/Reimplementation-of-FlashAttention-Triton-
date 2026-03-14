"""
Weight loader — maps HuggingFace Llama weights to our model.

Handles downloading the model from HuggingFace Hub and converting
the state dict keys from HuggingFace format to our module format.
"""

import torch
from pathlib import Path
from .model import LlamaModel, LlamaConfig


# HuggingFace key → our key mapping
_KEY_MAP = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Per-layer mappings (format string with layer index)
_LAYER_KEY_MAP = {
    "model.layers.{i}.self_attn.q_proj.weight":    "layers.{i}.attention.q_proj.weight",
    "model.layers.{i}.self_attn.k_proj.weight":    "layers.{i}.attention.k_proj.weight",
    "model.layers.{i}.self_attn.v_proj.weight":    "layers.{i}.attention.v_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight":    "layers.{i}.attention.o_proj.weight",
    "model.layers.{i}.mlp.gate_proj.weight":       "layers.{i}.mlp.gate_proj.weight",
    "model.layers.{i}.mlp.up_proj.weight":         "layers.{i}.mlp.up_proj.weight",
    "model.layers.{i}.mlp.down_proj.weight":       "layers.{i}.mlp.down_proj.weight",
    "model.layers.{i}.input_layernorm.weight":     "layers.{i}.input_layernorm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.post_attention_layernorm.weight",
}


def _map_key(hf_key: str, num_layers: int) -> str | None:
    """Map a single HuggingFace key to our key."""
    if hf_key in _KEY_MAP:
        return _KEY_MAP[hf_key]

    for i in range(num_layers):
        for hf_pattern, our_pattern in _LAYER_KEY_MAP.items():
            hf_resolved = hf_pattern.format(i=i)
            if hf_key == hf_resolved:
                return our_pattern.format(i=i)

    return None  # Skip unknown keys


def load_hf_weights(
    model: LlamaModel,
    model_name: str = "meta-llama/Llama-3.2-1B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LlamaModel:
    """
    Download and load HuggingFace Llama weights into our model.

    Args:
        model: Our LlamaModel instance (uninitialised weights)
        model_name: HuggingFace model ID
        device: Target device
        dtype: Target dtype

    Returns:
        Model with loaded weights
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_safetensors
    except ImportError:
        raise ImportError(
            "Install required packages: pip install huggingface_hub safetensors"
        )

    print(f"📥 Downloading {model_name}...")
    model_dir = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json"],
    )
    model_dir = Path(model_dir)
    print(f"   Cached at: {model_dir}")

    # Load safetensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    print(f"📂 Loading {len(safetensor_files)} safetensor file(s)...")
    hf_state_dict = {}
    for f in safetensor_files:
        hf_state_dict.update(load_safetensors(str(f)))

    # Map keys
    num_layers = model.config.num_hidden_layers
    our_state_dict = {}
    skipped = []

    for hf_key, tensor in hf_state_dict.items():
        our_key = _map_key(hf_key, num_layers)
        if our_key is None:
            skipped.append(hf_key)
            continue
        our_state_dict[our_key] = tensor.to(dtype)

    if skipped:
        print(f"   ⚠️  Skipped {len(skipped)} HF keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Handle tied embeddings (lm_head.weight == embed_tokens.weight)
    if "lm_head.weight" not in our_state_dict and model.config.tie_word_embeddings:
        our_state_dict["lm_head.weight"] = our_state_dict["embed_tokens.weight"]
        print("   🔗 Tied lm_head weights to embed_tokens")

    # Load into model
    missing, unexpected = model.load_state_dict(our_state_dict, strict=False)
    if missing:
        print(f"   ⚠️  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {unexpected[:5]}")

    model = model.to(device=device, dtype=dtype)
    print(f"✅ Loaded {len(our_state_dict)} tensors → {device} ({dtype})")

    return model
