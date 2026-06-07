import math
from modelinfo.calculator import calculate_footprint, _get_bytes_per_param

def test_quantization_byte_multipliers():
    """Verify block quantization ratios are correct per modern GGUF statistics."""
    assert _get_bytes_per_param("Q8") == 1.06
    assert _get_bytes_per_param("Q6") == 0.82
    assert _get_bytes_per_param("Q5") == 0.68
    assert _get_bytes_per_param("Q4") == 0.58
    assert _get_bytes_per_param("Q3") == 0.43
    assert _get_bytes_per_param("Q2") == 0.28
    
    # Baseline checks
    assert _get_bytes_per_param("BF16") == 2.0
    assert _get_bytes_per_param("F16") == 2.0
    assert _get_bytes_per_param("F32") == 4.0

def test_calculate_footprint_vram():
    """Ensure the footprint calculator accurately scales dimensions and types."""
    mock_tensors = {
        "model.layers.0.self_attn.q_proj.weight": {
            "shape": [4096, 4096], 
            "dtype": "BF16"
        },
        "model.embed_tokens.weight": {
            "shape": [32000, 4096],
            "dtype": "F32"
        }
    }
    
    # Calculate footprint without context
    footprint = calculate_footprint(mock_tensors)
    
    # 4096 * 4096 = 16,777,216 params
    # 32000 * 4096 = 131,072,000 params
    # Total = 147,849,216
    assert footprint["total_params"] == 147849216
    
    # Base memory:
    # 16,777,216 * 2.0 (BF16) = 33,554,432
    # 131,072,000 * 4.0 (F32) = 524,288,000
    # Total = 557,842,432 bytes
    assert footprint["base_memory_bytes"] == 557842432

def test_dynamic_kv_cache():
    """Ensure KV cache overhead scales correctly with context window."""
    mock_tensors = {
        "model.layers.0.self_attn.k_proj.weight": {
            "shape": [1024, 4096], # KV dim = 1024
            "dtype": "BF16"
        },
        "model.layers.1.self_attn.k_proj.weight": {
            "shape": [1024, 4096],
            "dtype": "BF16"
        }
    }
    # 2 layers, kv_dim=1024
    # Formula: 2 * Layers * KV_Dim * Context * Batch * 2
    # Context=1000, Batch=1
    # 2 * 2 * 1024 * 1000 * 1 * 2 = 8,192,000 bytes
    
    footprint = calculate_footprint(mock_tensors, context_length=1000, batch_size=1)
    
    assert footprint["num_layers"] == 2
    assert footprint["kv_dim"] == 1024
    assert footprint["kv_cache_bytes"] == 8192000

def test_safetensors_config_fallback():
    """Verify that architecture extraction correctly parses a config dictionary for SafeTensors."""
    from modelinfo.architecture import extract_architecture
    
    tensors = {
        "model.layers.0.qkv_proj.weight": {
            "shape": [6144, 4096],
            "dtype": "F16"
        }
    }
    
    config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096
    }
    
    num_layers, kv_dim, is_estimate = extract_architecture(tensors, config=config)
    
    assert num_layers == 32
    assert kv_dim == 1024
    assert is_estimate is False

def test_kv_cache_is_fp16():
    """Verify that KV cache is always calculated using 2.0 bytes (FP16), even for Q4 base models."""
    tensors = {
        "model.layers.0.attn.weight": {"shape": [4096, 4096], "dtype": "Q4"},
        "model.layers.0.attn.k.weight": {"shape": [1024, 4096], "dtype": "Q4"},
    }
    
    footprint = calculate_footprint(tensors, context_length=8192)
    
    assert footprint["kv_cache_bytes"] == 33554432
    assert footprint["primary_dtype"] == "Q4"

def test_framework_overhead_included():
    """Verify that CUDA context and activation overhead is correctly included."""
    tensors = {
        "model.layers.0.attn.weight": {"shape": [1024, 1024], "dtype": "F16"}
    }
    footprint = calculate_footprint(tensors)
    
    assert "overhead_bytes" in footprint
    assert footprint["overhead_bytes"] == 600 * 1024 * 1024
    assert footprint["total_memory_bytes"] == footprint["base_memory_bytes"] + footprint["kv_cache_bytes"] + footprint["overhead_bytes"]

def test_explicit_gguf_quantization_byte_multipliers():
    """Verify that explicit ggml_type enums are exactly mapped."""
    assert _get_bytes_per_param("Q8_0") == 1.0625
    assert _get_bytes_per_param("Q4_K") == 0.59375
    assert _get_bytes_per_param("IQ2_XXS") == 0.28125
    assert _get_bytes_per_param("F8_E5M2") == 1.0

def test_topology_penalties():
    """Verify multi-GPU distributed overhead logic."""
    tensors = {
        "model.layers.0.attn.weight": {"shape": [1024, 1024], "dtype": "F16"} # Base: 2,097,152 bytes
    }
    # NVLink (4%)
    fp_nvlink = calculate_footprint(tensors, gpu_count=2, topology="nvlink", strategy="tp")
    assert fp_nvlink["penalty_percentage"] == 0.04
    assert fp_nvlink["overhead_bytes"] == (2 * 600 * 1024 * 1024) + (2097152 * 0.04)

    # PCIe3 (20%)
    fp_pcie3 = calculate_footprint(tensors, gpu_count=4, topology="pcie3", strategy="tp")
    assert fp_pcie3["penalty_percentage"] == 0.20
    assert fp_pcie3["overhead_bytes"] == (4 * 600 * 1024 * 1024) + (2097152 * 0.20)

def test_strategy_pp():
    """Verify Pipeline Parallelism incurs 0 distributed overhead."""
    tensors = {
        "model.layers.0.attn.weight": {"shape": [1024, 1024], "dtype": "F16"}
    }
    fp_pp = calculate_footprint(tensors, gpu_count=4, topology="pcie3", strategy="pp")
    assert fp_pp["penalty_percentage"] == 0.0
    assert fp_pp["overhead_bytes"] == (4 * 600 * 1024 * 1024)

def test_vllm_subtractive_math():
    """Verify the subtractive vLLM serving capacity engine calculates exact tokens."""
    tensors = {
        "model.layers.0.attn.weight": {"shape": [1024, 1024], "dtype": "F16"} # Base: 2MB
    }
    config = {
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 1024
    }
    # 24GB VRAM. 90% util = 21.6GB. Base weights = 2MB. Remaining = ~21.59GB.
    # Bytes per token: 2 (FP16) * 10 (layers) * 1024 (kv_dim) * 2 = 40960 bytes
    gpu_vram = 24.0 * 1024**3
    
    fp_vllm = calculate_footprint(tensors, config=config, is_vllm=True, gpu_vram_bytes=gpu_vram, gpu_util=0.9, gpu_count=1)
    
    metrics = fp_vllm["vllm_metrics"]
    assert "usable_vram" in metrics
    assert metrics["usable_vram"] == gpu_vram * 0.9
    assert metrics["static_weights"] == 2097152
    assert metrics["paged_kv_pool"] == metrics["usable_vram"] - metrics["static_weights"]
    
    bytes_per_token = 40960
    expected_capacity = math.floor(metrics["paged_kv_pool"] / bytes_per_token)
    assert metrics["max_serving_capacity"] == expected_capacity
