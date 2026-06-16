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
