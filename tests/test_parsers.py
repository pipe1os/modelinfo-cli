import os
import pytest
from modelinfo.parsers.huggingface import _get_hf_endpoint
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

def test_gguf_parser_metadata():
    """Verify that the GGUF parser extracts the global metadata bypass."""
    from modelinfo.parsers.gguf import parse_gguf_header
    from modelinfo.architecture import identify_architecture_name
    
    mock_path = os.path.join(FIXTURES_DIR, "mock_model.gguf")
    tensors = parse_gguf_header(mock_path)
    
    assert "__metadata__" in tensors
    assert tensors["__metadata__"]["general.architecture"] == "qwen2"
    
    # Verify the architecture bypass parses it to titlecase and prevents "Unknown Architecture"
    arch_name = identify_architecture_name(tensors, num_layers=1)
    assert arch_name == "Qwen2 (1 transformer layers)"




def test_hf_endpoint_valid_https(monkeypatch):
    """Valid https:// endpoint is accepted."""
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    assert _get_hf_endpoint() == "https://huggingface.co"


def test_hf_endpoint_default_https(monkeypatch):
    """Default endpoint when HF_ENDPOINT is not set."""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    endpoint = _get_hf_endpoint()
    assert endpoint == "https://huggingface.co"


def test_hf_endpoint_rejects_http(monkeypatch):
    """http:// scheme is rejected with ValueError."""
    monkeypatch.setenv("HF_ENDPOINT", "http://localhost:8080")
    with pytest.raises(ValueError, match="must use https:// scheme"):
        _get_hf_endpoint()


def test_hf_endpoint_rejects_empty(monkeypatch):
    """Empty string is rejected with ValueError."""
    monkeypatch.setenv("HF_ENDPOINT", "")
    with pytest.raises(ValueError):
        _get_hf_endpoint()


def test_hf_endpoint_rejects_no_hostname(monkeypatch):
    """URL without a hostname is rejected with ValueError."""
    monkeypatch.setenv("HF_ENDPOINT", "https:///repo")
    with pytest.raises(ValueError, match="must include a valid hostname"):
        _get_hf_endpoint()


def test_remote_gguf_parsing_single(monkeypatch):
    """Test remote GGUF parsing when a single GGUF is found in the repository."""
    import json
    from modelinfo.parsers import huggingface
    
    def fake_make_request(url, headers=None, limit=None, timeout=10.0):
        if "/api/models/" in url:
            return json.dumps({
                "siblings": [
                    {"rfilename": "model-q4.gguf", "size": 1000000000}
                ]
            }).encode("utf-8")
        elif "model-q4.gguf" in url:
            import struct
            header = b"GGUF" + struct.pack("<IQQ", 2, 0, 0)
            return header
        raise ValueError(f"Unexpected url: {url}")
        
    monkeypatch.setattr(huggingface, "_make_request", fake_make_request)
    
    tensors, config, format_name, disk_size = huggingface.fetch_huggingface_repo("org/model-gguf")
    
    assert format_name == "GGUF"
    assert disk_size == 1000000000.0
    assert tensors.get("__metadata__") == {}


def test_remote_gguf_parsing_group(monkeypatch):
    """Test remote GGUF parsing when multiple GGUF files are present in the repository."""
    import json
    from modelinfo.parsers import huggingface
    
    def fake_make_request(url, headers=None, limit=None, timeout=10.0):
        if "/api/models/" in url:
            return json.dumps({
                "siblings": [
                    {"rfilename": "model-q4.gguf", "size": 1000000000},
                    {"rfilename": "model-q8.gguf", "size": 2000000000}
                ]
            }).encode("utf-8")
        elif "model-q4.gguf" in url:
            import struct
            header = b"GGUF" + struct.pack("<IQQ", 2, 0, 0)
            return header
        raise ValueError(f"Unexpected url: {url}")
        
    monkeypatch.setattr(huggingface, "_make_request", fake_make_request)
    
    tensors, config, format_name, disk_size = huggingface.fetch_huggingface_repo("org/model-gguf")
    
    assert format_name == "GGUF_group"
    assert disk_size == 0.0
    assert "gguf_variants" in tensors["__metadata__"]
    assert len(tensors["__metadata__"]["gguf_variants"]) == 2


def test_remote_gguf_parsing_explicit(monkeypatch):
    """Test remote GGUF parsing when the user targets a specific GGUF file in the repo id."""
    import json
    from modelinfo.parsers import huggingface
    
    called_gguf = []
    def fake_make_request(url, headers=None, limit=None, timeout=10.0):
        if "/api/models/" in url:
            return json.dumps({
                "siblings": [
                    {"rfilename": "model-q4.gguf", "size": 1000000000},
                    {"rfilename": "model-q8.gguf", "size": 2000000000}
                ]
            }).encode("utf-8")
        elif "model-q8.gguf" in url:
            called_gguf.append("q8")
            import struct
            header = b"GGUF" + struct.pack("<IQQ", 2, 0, 0)
            return header
        raise ValueError(f"Unexpected url: {url}")
        
    monkeypatch.setattr(huggingface, "_make_request", fake_make_request)
    
    tensors, config, format_name, disk_size = huggingface.fetch_huggingface_repo("org/model-gguf/model-q8.gguf")
    
    assert format_name == "GGUF"
    assert disk_size == 2000000000.0
    assert called_gguf == ["q8"]


def test_remote_gguf_parsing_unauthorized(monkeypatch):
    """Test remote parsing raises PermissionError for gated/unauthorized (401) model repositories."""
    import urllib.error
    from modelinfo.parsers import huggingface
    
    def fake_make_request(url, headers=None, limit=None, timeout=10.0):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        
    monkeypatch.setattr(huggingface, "_make_request", fake_make_request)
    
    import pytest
    with pytest.raises(PermissionError) as exc_info:
        huggingface.fetch_huggingface_repo("org/gated-model")
    assert "Gated/Private Model" in str(exc_info.value)


def test_remote_gguf_parsing_not_found(monkeypatch):
    """Test remote parsing raises FileNotFoundError for missing (404) model repositories."""
    import urllib.error
    from modelinfo.parsers import huggingface
    
    def fake_make_request(url, headers=None, limit=None, timeout=10.0):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        
    monkeypatch.setattr(huggingface, "_make_request", fake_make_request)
    
    import pytest
    with pytest.raises(FileNotFoundError) as exc_info:
        huggingface.fetch_huggingface_repo("org/nonexistent-model")
    assert "Could not find repository on Hugging Face" in str(exc_info.value)

def test_safetensors_sharded_with_hyphens(tmp_path):
    """Test safetensors parser sharded index path resolution when filename contains hyphens."""
    import struct
    import json
    
    index_file = tmp_path / "mock-llama-3-8b.safetensors.index.json"
    shard_file = tmp_path / "mock-llama-3-8b-00001-of-00002.safetensors"
    
    index_data = {
        "weight_map": {
            "model.embed_tokens.weight": "mock-llama-3-8b-00001-of-00002.safetensors"
        }
    }
    index_file.write_text(json.dumps(index_data), encoding="utf-8")
    
    header_data = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [32000, 4096],
            "data_offsets": [0, 262144000]
        }
    }
    header_json = json.dumps(header_data).encode("utf-8")
    header_len = len(header_json)
    
    with open(shard_file, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        
    tensors = parse_safetensors_header(str(shard_file))
    
    assert tensors.get("__metadata__", {}).get("is_sharded") is True
    assert tensors.get("__metadata__", {}).get("total_shards") == 1
    assert tensors.get("__metadata__", {}).get("missing_shards") == 0
    assert "model.embed_tokens.weight" in tensors
    assert tensors["model.embed_tokens.weight"]["dtype"] == "BF16"


def test_remote_shard_download_failure(monkeypatch):
    """Test remote sharded safetensors parsing when one of the shard downloads fails."""
    import json
    import struct
    import urllib.error
    from modelinfo.parsers import huggingface

    def fake_make_request(url, headers=None, limit=None, timeout=10.0):
        if "/api/models/" in url:
            return json.dumps({
                "siblings": [
                    {"rfilename": "model.safetensors.index.json"},
                    {"rfilename": "model-00001-of-00002.safetensors"},
                    {"rfilename": "model-00002-of-00002.safetensors"}
                ]
            }).encode("utf-8")
        elif "model.safetensors.index.json" in url:
            return json.dumps({
                "metadata": {"total_size": 2000000000},
                "weight_map": {
                    "layer1.weight": "model-00001-of-00002.safetensors",
                    "layer2.weight": "model-00002-of-00002.safetensors"
                }
            }).encode("utf-8")
        elif "model-00001-of-00002.safetensors" in url:
            header_json = json.dumps({"layer1.weight": {"dtype": "BF16", "shape": [1024, 1024]}}).encode("utf-8")
            return struct.pack("<Q", len(header_json)) + header_json
        elif "model-00002-of-00002.safetensors" in url:
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", {}, None)
        raise ValueError(f"Unexpected url: {url}")

    monkeypatch.setattr(huggingface, "_make_request", fake_make_request)

    tensors, config, format_name, disk_size = huggingface.fetch_huggingface_repo(
        "org/sharded-safetensors-model", fetch_tensors=True
    )

    assert format_name == "SafeTensors"
    assert disk_size == 2000000000.0
    assert tensors["__metadata__"]["missing_shards"] == 1
    assert tensors["__metadata__"]["total_shards"] == 2
    assert tensors["__metadata__"]["is_sharded"] is True
    assert "layer1.weight" in tensors
    assert "layer2.weight" not in tensors
