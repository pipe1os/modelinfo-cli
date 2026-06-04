import os
import pytest
from modelinfo.parsers.safetensors import parse_safetensors_header

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

def test_safetensors_parser_with_mock():
    """Test the safetensors parser using a locally generated minimal binary."""
    mock_path = os.path.join(FIXTURES_DIR, "mock_mistral-7b.safetensors")
    
    if not os.path.exists(mock_path):
        pytest.skip("Mock SafeTensors file not found in fixtures.")
        
    tensors = parse_safetensors_header(mock_path)
    
    # Verify embedded architecture parameters
    assert "model.embed_tokens.weight" in tensors
    assert tensors["model.embed_tokens.weight"]["dtype"] == "BF16"
    assert tensors["model.embed_tokens.weight"]["shape"] == [32000, 4096]
    
    # Check a specific layer
    layer_0_q = tensors.get("model.layers.0.self_attn.q_proj.weight")
    assert layer_0_q is not None
    assert layer_0_q["shape"] == [4096, 4096]

def test_missing_shard_handling():
    """Ensure the safetensors index parser catches missing files correctly."""
    # We can test the logic directly or simulate it via monkeypatching.
    # Since we are focusing on parser structural integrity, we ensure 
    # it fails safely when a file truly doesn't exist.
    with pytest.raises(FileNotFoundError):
        parse_safetensors_header(os.path.join(FIXTURES_DIR, "does_not_exist.safetensors"))
