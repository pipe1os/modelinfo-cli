import json

import pytest

from modelinfo import cli

GB = 1024**3


def make_info(**overrides):
    info = {
        "format_name": "GGUF",
        "arch_name": "Llama (2 transformer layers)",
        "tensor_count": 1,
        "footprint": {
            "total_params": 9007199254740993,
            "base_memory_bytes": 4 * GB,
            "kv_cache_bytes": GB,
            "overhead_bytes": GB,
            "total_memory_bytes": 6 * GB,
            "num_layers": 2,
            "kv_dim": 1024,
            "primary_dtype": "Q4_K",
            "kv_is_estimate": False,
            "penalty_percentage": 0.0,
            "vllm_metrics": {},
        },
        "disk_size": 4 * GB,
        "context_length": 8192,
        "is_default_context": True,
        "tensors": {
            "__metadata__": {
                "general.architecture": "llama",
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.tokens": ["a"] * 1000,
                "tokenizer.ggml.merges": ["a b"] * 1000,
                "llama.context_length": 8192,
            },
            "blk.0.attn_k.weight": {"shape": [4096, 1024], "dtype": "Q4_K"},
        },
        "max_context": 8192,
        "is_lazy": False,
        "gpu_count": 1,
        "topology": "pcie4",
        "strategy": "tp",
        "is_vllm": False,
        "gpu_vram_gb": 0.0,
        "gpu_util": 0.9,
    }
    info.update(overrides)
    return info


@pytest.fixture
def analyzed(monkeypatch):
    info = make_info()
    monkeypatch.setattr(cli, "analyze_model", lambda *a, **kw: info)
    monkeypatch.setattr(
        cli,
        "print_model_info",
        lambda **kw: pytest.fail("Unexpected terminal renderer"),
    )
    monkeypatch.setattr(
        cli,
        "print_compare_info",
        lambda *a, **kw: pytest.fail("Unexpected comparison renderer"),
    )
    return info


@pytest.fixture
def gpu(monkeypatch):
    from modelinfo import hardware

    def use(vram_gb):
        monkeypatch.setattr(
            hardware, "resolve_gpu", lambda target: ("Test GPU", vram_gb, 1)
        )

    return use


def test_json_flag_defaults_to_false():
    assert cli.parse_args(["model.gguf"]).json is False


def test_json_single_model_is_machine_readable(analyzed, capsys):
    assert cli.main(["model.gguf", "--json"]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert "\x1b" not in captured.out
    assert report["footprint"] == analyzed["footprint"]
    assert report["footprint"]["total_params"] == 9007199254740993
    assert report["format_name"] == "GGUF"
    assert "gpu_name" not in report
    assert "hardware_fit" not in report


def test_json_omits_tensor_table_and_tokenizer_arrays(analyzed, capsys):
    assert cli.main(["model.gguf", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "tensors" not in report
    assert report["metadata"] == {
        "general.architecture": "llama",
        "tokenizer.ggml.model": "gpt2",
        "llama.context_length": 8192,
    }


def test_json_tensors_flag_includes_tensor_table(analyzed, capsys):
    assert cli.main(["model.gguf", "--json", "--tensors"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["tensors"] == {
        "blk.0.attn_k.weight": {"shape": [4096, 1024], "dtype": "Q4_K"}
    }


def test_json_comparison_preserves_input_order_and_names(analyzed, capsys):
    assert cli.main(["first/model.gguf", "second/other.gguf", "--json"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in reports] == ["model.gguf", "other.gguf"]
    assert all(entry["info"]["footprint"] == analyzed["footprint"] for entry in reports)


@pytest.mark.parametrize(
    "vram_gb,expected", [(24.0, "yes"), (6.2, "warning"), (4.0, "no")]
)
def test_json_reports_hardware_fit_for_target_gpu(analyzed, gpu, capsys, vram_gb, expected):
    gpu(vram_gb)
    assert cli.main(["model.gguf", "--json", "--gpu", "test"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_name"] == "Test GPU"
    assert report["hardware_fit"] == expected


def test_json_hardware_fit_is_unknown_for_missing_shards(analyzed, gpu, capsys):
    gpu(24.0)
    analyzed["tensors"]["__metadata__"] = {"missing_shards": 1, "total_shards": 2}
    assert cli.main(["model.gguf", "--json", "--gpu", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["hardware_fit"] == "unknown"


def test_json_comparison_reports_fit_per_model(analyzed, gpu, capsys):
    gpu(24.0)
    assert cli.main(["a/one.gguf", "b/two.gguf", "--json", "--gpu", "test"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert [entry["info"]["hardware_fit"] for entry in reports] == ["yes", "yes"]


@pytest.mark.parametrize("pool,expected", [(GB, "yes"), (0.0, "no")])
def test_json_single_model_supports_vllm_metadata(analyzed, gpu, capsys, pool, expected):
    gpu(24.0)
    analyzed["is_vllm"] = True
    analyzed["footprint"]["vllm_metrics"] = {
        "max_serving_capacity": 1234,
        "paged_kv_pool": pool,
    }
    assert cli.main(["model.gguf", "--json", "--vllm"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["footprint"]["vllm_metrics"]["max_serving_capacity"] == 1234
    assert report["hardware_fit"] == expected


def test_json_gguf_group_lists_variant_totals_and_fits(analyzed, gpu, capsys):
    gpu(2.0)
    analyzed["format_name"] = "GGUF_group"
    analyzed["footprint"]["kv_cache_bytes"] = 0
    analyzed["tensors"]["__metadata__"] = {
        "repo_id": "org/model-gguf",
        "gguf_variants": [
            {"filename": "model-q4.gguf", "size": 0.5 * GB},
            {"filename": "model-q8.gguf", "size": 1.5 * GB},
        ],
    }
    assert cli.main(["org/model-gguf", "--json", "--gpu", "test"]) == 0
    report = json.loads(capsys.readouterr().out)
    overhead = 600 * 1024 * 1024
    assert report["variants"] == [
        {"filename": "model-q8.gguf", "size": 1.5 * GB,
         "total_vram_bytes": 1.5 * GB + overhead, "fits": "no"},
        {"filename": "model-q4.gguf", "size": 0.5 * GB,
         "total_vram_bytes": 0.5 * GB + overhead, "fits": "yes"},
    ]
    assert "hardware_fit" not in report


def test_comparison_vllm_error_keeps_stdout_clean(analyzed, gpu, capsys):
    gpu(24.0)
    assert cli.main(["a/one.gguf", "b/two.gguf", "--json", "--vllm"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does not currently support" in captured.err
