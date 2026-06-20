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
