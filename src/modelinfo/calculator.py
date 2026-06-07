import math
from typing import Any, Dict

from modelinfo.architecture import extract_architecture

DTYPE_BYTES = {
    "F64": 8.0,
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "F8": 1.0,
    "F8_E5M2": 1.0,
    "F8_E4M3": 1.0,
    "I64": 8.0,
    "I32": 4.0,
    "I16": 2.0,
    "I8": 1.0,
    "U64": 8.0,
    "U32": 4.0,
    "Q8_0": 1.0625,
    "Q8_1": 1.0625,
    "Q8_K": 1.0625,
    "Q6_K": 0.828125,
    "Q5_0": 0.6875,
    "Q5_1": 0.75,
    "Q5_K": 0.6875,
    "Q4_0": 0.5625,
    "Q4_1": 0.625,
    "Q4_K": 0.59375,
    "Q3_K": 0.4375,
    "Q2_K": 0.34375,
    "IQ4_NL": 0.53125,
    "IQ4_XS": 0.53125,
    "IQ3_S": 0.4375,
    "IQ3_XXS": 0.385,
    "IQ2_S": 0.3125,
    "IQ2_XS": 0.296875,
    "IQ2_XXS": 0.28125,
    "IQ1_M": 0.21875,
    "IQ1_S": 0.1953125,
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

def calculate_footprint(
    tensors: Dict[str, Any], 
    context_length: int = 0, 
    batch_size: int = 1, 
    config: Dict[str, Any] = None, 
    gpu_count: int = 1,
    topology: str = "pcie4",
    strategy: str = "tp",
    is_vllm: bool = False,
    gpu_vram_bytes: float = 0.0,
    gpu_util: float = 0.9
) -> Dict[str, Any]:
    """
    Calculate the memory footprint of a model based on its tensors and context length.
    """
    total_params = 0
    base_memory_bytes = 0.0
    dtype_counts: Dict[str, int] = {}
    
    is_lazy = tensors.get("__metadata__", {}).get("lazy_fetch", False)
    
    if is_lazy:
        base_memory_bytes = tensors.get("__metadata__", {}).get("total_size", 0.0)
        # Assume predominantly FP16/BF16 for modern Hub architectures
        primary_dtype = "BF16"
        dtype_counts[primary_dtype] = 1
        total_params = int(base_memory_bytes / 2.0)
    else:
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
        
    num_layers, kv_dim, is_estimate = extract_architecture(tensors, config)
    
    # Formula: 2 * Layers * (KV_Heads * Head_Dim) * Context_Length * Batch_Size * Bytes_per_param
    # Assume FP16 (2 bytes) for KV cache
    kv_cache_bytes = 2 * num_layers * kv_dim * context_length * batch_size * 2
    
    primary_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0] if dtype_counts else "Unknown"
    # Topology & Strategy Penalties
    penalty_percentage = 0.0
    if gpu_count > 1:
        if strategy == "pp":
            penalty_percentage = 0.0
        else: # strategy == "tp"
            if topology == "nvlink":
                penalty_percentage = 0.04
            elif topology == "pcie3":
                penalty_percentage = 0.20
            else: # pcie4
                penalty_percentage = 0.12
                
    distributed_overhead = base_memory_bytes * penalty_percentage if gpu_count > 1 else 0.0
    
    vllm_metrics = {}
    if is_vllm and gpu_vram_bytes > 0:
        usable_vram = gpu_vram_bytes * gpu_util
        remaining_vram = usable_vram - (base_memory_bytes + distributed_overhead)
        
        bytes_per_token = 2 * num_layers * kv_dim * 2
        
        max_serving_capacity = 0
        if remaining_vram > 0 and bytes_per_token > 0:
            max_serving_capacity = math.floor(remaining_vram / bytes_per_token)
            
        overhead_bytes = distributed_overhead
        total_memory_bytes = base_memory_bytes + overhead_bytes
        
        vllm_metrics = {
            "usable_vram": usable_vram,
            "static_weights": base_memory_bytes,
            "distributed_penalty": distributed_overhead,
            "paged_kv_pool": max(0.0, remaining_vram),
            "max_serving_capacity": max_serving_capacity
        }
    else:
        CUDA_CONTEXT_MB = 600 * gpu_count
        overhead_bytes = (CUDA_CONTEXT_MB * 1024 * 1024) + distributed_overhead
        total_memory_bytes = base_memory_bytes + kv_cache_bytes + overhead_bytes
    
    return {
        "total_params": total_params,
        "base_memory_bytes": base_memory_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "overhead_bytes": overhead_bytes,
        "total_memory_bytes": total_memory_bytes,
        "num_layers": num_layers,
        "kv_dim": kv_dim,
        "primary_dtype": primary_dtype,
        "kv_is_estimate": is_estimate,
        "penalty_percentage": penalty_percentage,
        "vllm_metrics": vllm_metrics
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
