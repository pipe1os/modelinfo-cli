import math
from typing import Any, Dict

from modelinfo.architecture import extract_architecture

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "Q8": 1.06,
    "Q6": 0.82,
    "Q5": 0.68,
    "Q4": 0.58,
    "Q3": 0.43,
    "Q2": 0.28,
}

def _get_bytes_per_param(dtype: str) -> float:
    """Return the size in bytes for a given data type."""
    return DTYPE_BYTES.get(dtype.upper(), 2.0)

def calculate_footprint(tensors: Dict[str, Any], context_length: int = 0, batch_size: int = 1) -> Dict[str, Any]:
    """
    Calculate the memory footprint of a model based on its tensors and context length.
    """
    total_params = 0
    base_memory_bytes = 0.0
    dtype_counts: Dict[str, int] = {}
    
    for name, metadata in tensors.items():
        if name == "__metadata__":
            continue
            
        shape = metadata.get("shape", [])
        if not shape:
            continue
            
        param_count = math.prod(shape)
        total_params += param_count
        
        dtype = metadata.get("dtype", "F16").upper()
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        bytes_per_param = _get_bytes_per_param(dtype)
        base_memory_bytes += param_count * bytes_per_param
        
    num_layers, kv_dim = extract_architecture(tensors)
    
    # Formula: 2 * Layers * (KV_Heads * Head_Dim) * Context_Length * Batch_Size * Bytes_per_param
    # Assume FP16 (2 bytes) for KV cache
    kv_cache_bytes = 2 * num_layers * kv_dim * context_length * batch_size * 2
    
    primary_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0] if dtype_counts else "Unknown"
    
    return {
        "total_params": total_params,
        "base_memory_bytes": base_memory_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "total_memory_bytes": base_memory_bytes + kv_cache_bytes,
        "num_layers": num_layers,
        "kv_dim": kv_dim,
        "primary_dtype": primary_dtype
    }

def format_bytes(size_bytes: float) -> str:
    """Format bytes into a human-readable string (e.g. GB)."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = max(0, min(len(units) - 1, math.floor(math.log(size_bytes, 1024))))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {units[i]}"

def format_params(count: int) -> str:
    """Format parameter count into a human-readable string (e.g. 7.2B)."""
    if count >= 1_000_000_000:
        return f"{count:,} ({count / 1_000_000_000:.1f}B)"
    elif count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.1f}M)"
    elif count >= 1_000:
        return f"{count:,} ({count / 1_000:.1f}K)"
    return f"{count:,}"
