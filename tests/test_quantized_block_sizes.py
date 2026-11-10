"""Expected byte lengths verified with sizeof GGML quantization block structures.

Reference: https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
This tests packed tensor storage, excluding GGUF file metadata and alignment.
"""
import pytest

from modelinfo.calculator import calculate_footprint


@pytest.mark.parametrize("dtype,block_size,block_bytes", [
    ('Q8_0', 32, 34),
    ('Q8_1', 32, 36),
    ('Q8_K', 256, 292),
    ('Q6_K', 256, 210),
    ('Q5_0', 32, 22),
    ('Q5_1', 32, 24),
    ('Q5_K', 256, 176),
    ('Q4_0', 32, 18),
    ('Q4_1', 32, 20),
    ('Q4_K', 256, 144),
    ('Q3_K', 256, 110),
    ('Q2_K', 256, 84),
    ('IQ4_NL', 32, 18),
    ('IQ4_XS', 256, 136),
    ('IQ3_S', 256, 110),
    ('IQ3_XXS', 256, 98),
    ('IQ2_S', 256, 82),
    ('IQ2_XS', 256, 74),
    ('IQ2_XXS', 256, 66),
    ('IQ1_M', 256, 56),
    ('IQ1_S', 256, 50),
])
def test_quantized_tensor_storage_matches_ggml_blocks(dtype, block_size, block_bytes):
    result = calculate_footprint({"weight": {"shape": [3, block_size], "dtype": dtype}})
    assert result["base_memory_bytes"] == 3 * block_bytes
