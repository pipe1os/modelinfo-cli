from typing import Any, Dict, Tuple

def extract_architecture(tensors: Dict[str, Any]) -> Tuple[int, int]:
    """
    Extracts the number of layers and KV cache dimension (kv_heads * head_dim)
    from tensor metadata.
    """
    layers = set()
    kv_dim = 0
    
    for name, metadata in tensors.items():
        if name == "__metadata__":
            continue
            
        parts = name.split(".")
        
        if "layers" in parts:
            idx = parts.index("layers")
            if len(parts) > idx + 1 and parts[idx+1].isdigit():
                layers.add(int(parts[idx+1]))
        elif "h" in parts:
            idx = parts.index("h")
            if len(parts) > idx + 1 and parts[idx+1].isdigit():
                layers.add(int(parts[idx+1]))

        if name.endswith("k_proj.weight") or name.endswith("attn.k.weight") or name.endswith("k_proj.w"):
            shape = metadata.get("shape", [])
            if len(shape) >= 2:
                # Typically [out_features, in_features], so out_features is shape[0]
                kv_dim = shape[0]

    return len(layers), kv_dim

def identify_architecture_name(tensors: Dict[str, Any], num_layers: int) -> str:
    """Attempt to identify the architecture family based on tensor names."""
    for name in tensors.keys():
        name_lower = name.lower()
        if "llama" in name_lower:
            return f"Llama ({num_layers} transformer layers)" if num_layers else "Llama"
        if "mistral" in name_lower:
            return f"Mistral ({num_layers} transformer layers)" if num_layers else "Mistral"
        if "qwen" in name_lower:
            return f"Qwen ({num_layers} transformer layers)" if num_layers else "Qwen"
        
    return f"Generic Transformer ({num_layers} layers)" if num_layers > 0 else "Unknown Architecture"
