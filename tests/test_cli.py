import pytest

import modelinfo.cli as cli
from modelinfo import __version__
from modelinfo.cli import parse_args


def test_version_flag_prints_installed_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--version"])

    assert exc_info.value.code == 0
    assert f"modelinfo {__version__}" in capsys.readouterr().out


def test_batch_size_flag_defaults_to_one():
    args = parse_args(["model.gguf"])

    assert args.batch_size == 1


def test_batch_size_flag_accepts_integer():
    args = parse_args(["--batch-size", "4", "model.gguf"])

    assert args.batch_size == 4


def test_batch_size_flag_rejects_zero():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--batch-size", "0", "model.gguf"])

    assert exc_info.value.code == 2


def test_batch_size_flag_rejects_negative():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--batch-size", "-1", "model.gguf"])

    assert exc_info.value.code == 2


def test_timeout_flag_defaults_to_ten_seconds():
    args = parse_args(["model.gguf"])

    assert args.timeout == 10.0


def test_timeout_flag_accepts_float():
    args = parse_args(["--timeout", "30.5", "model.gguf"])

    assert args.timeout == 30.5


def test_timeout_flag_rejects_zero():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--timeout", "0", "model.gguf"])

    assert exc_info.value.code == 2


def test_timeout_flag_rejects_negative():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--timeout", "-1", "model.gguf"])

    assert exc_info.value.code == 2


def test_timeout_flag_rejects_nan():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--timeout", "nan", "model.gguf"])

    assert exc_info.value.code == 2


def test_timeout_flag_rejects_inf():
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--timeout", "inf", "model.gguf"])

    assert exc_info.value.code == 2


def test_analyze_model_passes_batch_size_to_footprint(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"mock")
    captured = {}

    def fake_parse_gguf_header(file_path):
        assert file_path == str(model_path)
        return {
            "model.layers.0.self_attn.k_proj.weight": {"shape": [1, 1], "dtype": "F16"}
        }

    def fake_calculate_footprint(tensors, *, context_length, batch_size, **kwargs):
        captured["batch_size"] = batch_size
        captured["context_length"] = context_length
        return {
            "total_params": 1,
            "base_memory_bytes": 2.0,
            "kv_cache_bytes": float(batch_size),
            "overhead_bytes": 0.0,
            "total_memory_bytes": 2.0 + batch_size,
            "num_layers": 1,
            "kv_dim": 1,
            "primary_dtype": "F16",
            "kv_is_estimate": False,
            "penalty_percentage": 0.0,
            "vllm_metrics": {},
        }

    monkeypatch.setattr(cli, "parse_gguf_header", fake_parse_gguf_header)
    monkeypatch.setattr(cli, "calculate_footprint", fake_calculate_footprint)
    monkeypatch.setattr(
        cli, "identify_architecture_name", lambda tensors, num_layers, config: "Mock"
    )

    info = cli.analyze_model(str(model_path), context_override=128, batch_size=4)

    assert captured == {"batch_size": 4, "context_length": 128}
    assert info["footprint"]["kv_cache_bytes"] == 4.0


def test_analyze_model_passes_timeout_to_huggingface(monkeypatch):
    captured = {}

    def fake_exists(path):
        return False

    def fake_fetch(repo_id, *, fetch_tensors, timeout):
        captured["repo_id"] = repo_id
        captured["fetch_tensors"] = fetch_tensors
        captured["timeout"] = timeout
        return (
            {
                "model.layers.0.self_attn.k_proj.weight": {
                    "shape": [1, 1],
                    "dtype": "F16",
                }
            },
            None,
            "SafeTensors",
            7.0,
        )

    def fake_calculate_footprint(tensors, *, context_length, batch_size, **kwargs):
        return {
            "total_params": 1,
            "base_memory_bytes": 2.0,
            "kv_cache_bytes": 1.0,
            "overhead_bytes": 0.0,
            "total_memory_bytes": 3.0,
            "num_layers": 1,
            "kv_dim": 1,
            "primary_dtype": "F16",
            "kv_is_estimate": False,
            "penalty_percentage": 0.0,
            "vllm_metrics": {},
        }

    from modelinfo.parsers import huggingface

    monkeypatch.setattr(cli.os.path, "exists", fake_exists)
    monkeypatch.setattr(huggingface, "fetch_huggingface_repo", fake_fetch)
    monkeypatch.setattr(cli, "calculate_footprint", fake_calculate_footprint)
    monkeypatch.setattr(
        cli, "identify_architecture_name", lambda tensors, num_layers, config: "Mock"
    )

    cli.analyze_model(
        "org/model",
        context_override=128,
        fetch_tensors=True,
        timeout=22.5,
    )

    assert captured == {
        "repo_id": "org/model",
        "fetch_tensors": True,
        "timeout": 22.5,
    }


def test_analyze_model_gguf_group(monkeypatch):
    """Test that analyze_model correctly handles and propagates GGUF groups."""
    from modelinfo.parsers import huggingface
    
    def fake_exists(path):
        return False
        
    def fake_fetch(repo_id, *, fetch_tensors, timeout):
        tensors, _ = _get_mock_gguf_group_data()
        return tensors, None, "GGUF_group", 0.0
        
    monkeypatch.setattr(cli.os.path, "exists", fake_exists)
    monkeypatch.setattr(huggingface, "fetch_huggingface_repo", fake_fetch)
    
    def fake_calculate_footprint(*args, **kwargs):
        return {
            "total_params": 1000000,
            "base_memory_bytes": 2000000.0,
            "kv_cache_bytes": 1000000.0,
            "overhead_bytes": 600000.0,
            "total_memory_bytes": 3600000.0,
            "num_layers": 32,
            "kv_dim": 1024,
            "primary_dtype": "Q4_0",
            "kv_is_estimate": False,
            "penalty_percentage": 0.0,
            "vllm_metrics": {}
        }
    monkeypatch.setattr(cli, "calculate_footprint", fake_calculate_footprint)
    
    info = cli.analyze_model("org/model-gguf", context_override=128)
    
    assert info["format_name"] == "GGUF_group"
    assert info["tensors"]["__metadata__"]["repo_id"] == "org/model-gguf"
    assert len(info["tensors"]["__metadata__"]["gguf_variants"]) == 2


def _get_mock_gguf_group_data():
    tensors = {
        "__metadata__": {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "gguf_variants": [
                {"filename": "model-q4.gguf", "size": 1000000000},
                {"filename": "model-q8.gguf", "size": 2000000000}
            ],
            "repo_id": "org/model-gguf"
        }
    }
    footprint = {
        "total_params": 8000000000,
        "base_memory_bytes": 4000000000.0,
        "kv_cache_bytes": 1000000000.0,
        "overhead_bytes": 600000000.0,
        "total_memory_bytes": 5600000000.0,
        "num_layers": 32,
        "kv_dim": 1024,
        "primary_dtype": "Q4_0",
        "kv_is_estimate": False,
        "penalty_percentage": 0.0,
        "vllm_metrics": {}
    }
    return tensors, footprint


def test_print_model_info_gguf_group_no_gpu(capsys):
    """Test print_model_info renders comparison table without Fits column when no GPU target."""
    from modelinfo.ui import print_model_info
    tensors, footprint = _get_mock_gguf_group_data()
    print_model_info(
        format_name="GGUF_group",
        arch_name="Llama (32 layers)",
        tensor_count=0,
        footprint=footprint,
        disk_size=0.0,
        context_length=8192,
        is_default_context=True,
        tensors=tensors,
        max_context=32768,
        max_vram_gb=8.0,
        gpu_name=None
    )
    out, _ = capsys.readouterr()
    assert "model-q4.gguf" in out
    assert "model-q8.gguf" in out
    assert "Fits" not in out
    assert "Tip:" in out


def test_print_model_info_gguf_group_with_gpu(capsys):
    """Test print_model_info renders comparison table with Fits column when GPU target exists."""
    from modelinfo.ui import print_model_info
    tensors, footprint = _get_mock_gguf_group_data()
    print_model_info(
        format_name="GGUF_group",
        arch_name="Llama (32 layers)",
        tensor_count=0,
        footprint=footprint,
        disk_size=0.0,
        context_length=8192,
        is_default_context=True,
        tensors=tensors,
        max_context=32768,
        max_vram_gb=8.0,
        gpu_name="RTX4080"
    )
    out, _ = capsys.readouterr()
    assert "model-q4.gguf" in out
    assert "model-q8.gguf" in out
    assert "Fits" in out

def test_analyze_model_local_path_routing(monkeypatch):
    """Test that analyze_model treats paths starting with local prefix as local, raising an error instead of routing to Hugging Face."""
    from modelinfo.parsers import huggingface

    hf_fetched = []
    def fake_fetch(repo_id, *, fetch_tensors, timeout):
        hf_fetched.append(repo_id)
        return {}, None, "SafeTensors", 0.0

    monkeypatch.setattr(huggingface, "fetch_huggingface_repo", fake_fetch)

    # Test cases that should NOT hit Hugging Face
    local_paths = ["./missing.gguf", "../missing.safetensors", "/missing.bin", "~/missing.pt"]
    for path in local_paths:
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            cli.analyze_model(path, context_override=128)

    assert len(hf_fetched) == 0, f"Hugging Face fetch was triggered for local paths: {hf_fetched}"

    # Test cases that SHOULD hit Hugging Face
    remote_paths = ["meta-llama/Llama-2-7b-hf", "org/model"]
    for path in remote_paths:
        try:
            cli.analyze_model(path, context_override=128)
        except Exception:
            # We don't care if calculation fails later because of empty dict from fake_fetch,
            # we just care that it triggers fetch_huggingface_repo.
            pass

    assert hf_fetched == remote_paths


def test_cli_strips_trailing_slashes_from_model_paths(monkeypatch):
    captured_paths = []
    
    def fake_analyze_model(file_path, *args, **kwargs):
        captured_paths.append(file_path)
        return {
            "format_name": "GGUF",
            "arch_name": "Llama",
            "tensor_count": 10,
            "footprint": {
                "total_params": 100,
                "base_memory_bytes": 200,
                "kv_cache_bytes": 100,
                "overhead_bytes": 50,
                "total_memory_bytes": 350,
                "num_layers": 1,
            },
            "disk_size": 200,
            "context_length": 128,
            "is_default_context": True,
            "tensors": {},
            "max_context": 512,
            "is_lazy": False,
            "gpu_count": 1,
            "topology": "pcie4",
            "strategy": "tp",
            "is_vllm": False,
            "gpu_vram_gb": 0.0,
            "gpu_util": 0.9,
        }

    monkeypatch.setattr(cli, "analyze_model", fake_analyze_model)
    monkeypatch.setattr(cli, "print_compare_info", lambda models, max_vram, gpu_name: None)
    monkeypatch.setattr(cli, "print_model_info", lambda *args, **kwargs: None)

    # Test single model path with trailing slash
    cli.main(["meta-llama/Llama-2-7b-hf/"])
    assert captured_paths == ["meta-llama/Llama-2-7b-hf"]

    captured_paths.clear()

    # Test multiple model paths with trailing slashes (side-by-side comparison)
    cli.main(["meta-llama/Llama-2-7b-hf/", "mistralai/Mistral-7B-v0.1/"])
    assert captured_paths == ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"]
