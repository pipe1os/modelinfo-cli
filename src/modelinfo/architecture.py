import os
import json
from typing import Any, Dict, Tuple

def extract_architecture(tensors: Dict[str, Any], config: Dict[str, Any] = None) -> Tuple[int, int, bool]:
    """
    Extracts the number of layers and KV cache dimension (kv_heads * head_dim).
    Returns (num_layers, kv_dim, is_estimate).
    """
    num_layers = 0
    kv_dim = 0
    is_estimate = False
    
    metadata = tensors.get("__metadata__", {})
    gen_arch = metadata.get("general.architecture")
    
    # 1. Attempt explicit GGUF metadata
    if gen_arch:
        arch_str = str(gen_arch)
        num_layers = metadata.get(f"{arch_str}.block_count", 0)
        kv_heads = metadata.get(f"{arch_str}.attention.head_count_kv", 0)
        
        key_length = metadata.get(f"{arch_str}.attention.key_length")
        if not key_length:
            embed_len = metadata.get(f"{arch_str}.embedding_length", 0)
            q_heads = metadata.get(f"{arch_str}.attention.head_count", 1)
            if q_heads > 0:
                key_length = embed_len // q_heads
            else:
                key_length = 0
                
        if kv_heads > 0 and key_length > 0:
            kv_dim = kv_heads * key_length
            if num_layers > 0:
                return num_layers, kv_dim, False

    # 2. Attempt explicit SafeTensors config.json
    if config:
        num_layers = config.get("num_hidden_layers", 0)
        num_attention_heads = config.get("num_attention_heads", 1)
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        hidden_size = config.get("hidden_size", 0)
        
        if num_attention_heads > 0:
            head_dim = hidden_size // num_attention_heads
            kv_dim = num_key_value_heads * head_dim
            if num_layers > 0 and kv_dim > 0:
                return num_layers, kv_dim, False

    # 3. Fallback to shape guessing
    layers_set = set()
    found_fused = False
    found_k_proj = False
    
    for name, meta in tensors.items():
        if name == "__metadata__":
            continue
            
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            if len(parts) > idx + 1 and parts[idx+1].isdigit():
                layers_set.add(int(parts[idx+1]))
        elif "h" in parts:
            idx = parts.index("h")
            if len(parts) > idx + 1 and parts[idx+1].isdigit():
                layers_set.add(int(parts[idx+1]))

        if name.endswith("k_proj.weight") or name.endswith("attn.k.weight") or name.endswith("k_proj.w"):
            found_k_proj = True
            shape = meta.get("shape", [])
            if len(shape) >= 2:
                kv_dim = shape[0]
                
        if "qkv_proj.weight" in name or "c_attn.weight" in name:
            found_fused = True
            if not found_k_proj:
                shape = meta.get("shape", [])
                if len(shape) >= 2:
                    kv_dim = shape[0] // 3

    num_layers = len(layers_set)
    if found_fused and not found_k_proj and kv_dim > 0:
        is_estimate = True
        
    return num_layers, kv_dim, is_estimate

def identify_architecture_name(tensors: Dict[str, Any], num_layers: int) -> str:
    """Attempt to identify the architecture family based on tensor names or metadata."""
    metadata = tensors.get("__metadata__", {})
    gen_arch = metadata.get("general.architecture")
    
    if gen_arch:
        arch_title = str(gen_arch).title()
        return f"{arch_title} ({num_layers} transformer layers)" if num_layers else arch_title
        
    for name in tensors.keys():
        if name == "__metadata__":
            continue
            
        name_lower = name.lower()
        if "llama" in name_lower:
            return f"Llama ({num_layers} transformer layers)" if num_layers else "Llama"
        if "mistral" in name_lower:
            return f"Mistral ({num_layers} transformer layers)" if num_layers else "Mistral"
        if "qwen" in name_lower:
            return f"Qwen ({num_layers} transformer layers)" if num_layers else "Qwen"
        
    return f"Generic Transformer ({num_layers} layers)" if num_layers > 0 else "Unknown Architecture"
