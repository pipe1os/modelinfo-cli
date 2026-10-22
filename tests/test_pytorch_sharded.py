"""Sharded PyTorch checkpoints (pytorch_model.bin.index.json); no torch needed."""

import json
import pickle
import zipfile

import pytest

import modelinfo.cli as cli
from modelinfo.parsers.pytorch import parse_pytorch_header, parse_pytorch_index


def write_pytorch_archive(path, tensors):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", pickle.dumps(tensors, protocol=2))


def write_sharded_model(tmp_path):
    shard_one = tmp_path / "pytorch_model-00001-of-00002.bin"
    shard_two = tmp_path / "pytorch_model-00002-of-00002.bin"
    write_pytorch_archive(shard_one, {"a": {"shape": [2], "dtype": "F16"}})
    write_pytorch_archive(shard_two, {"b": {"shape": [3], "dtype": "F32"}})
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(
        json.dumps({"weight_map": {"a": shard_one.name, "b": shard_two.name}})
    )
    return index, shard_one, shard_two


def test_pytorch_index_merges_shards_and_reports_missing_files(tmp_path):
    shard = tmp_path / "pytorch_model-00001-of-00002.bin"
    write_pytorch_archive(
        shard,
        {"a": {"shape": [2], "dtype": "F16"}, "b": {"shape": [3], "dtype": "F32"}},
    )
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": shard.name,
                    "b": shard.name,
                    "c": "pytorch_model-00002-of-00002.bin",
                }
            }
        )
    )
    result = parse_pytorch_index(str(index))
    assert result["a"] == {"shape": [2], "dtype": "F16"}
    assert result["b"] == {"shape": [3], "dtype": "F32"}
    assert "c" not in result
    assert result["__metadata__"] == {
        "missing_shards": 1,
        "total_shards": 2,
        "is_sharded": True,
        "disk_size": shard.stat().st_size,
    }


def test_pytorch_index_selects_only_assigned_tensors(tmp_path):
    shard = tmp_path / "shard.bin"
    write_pytorch_archive(
        shard,
        {
            "weight": {"shape": [3], "dtype": "F16"},
            "unused": {"shape": [1000], "dtype": "F32"},
        },
    )
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    result = parse_pytorch_index(str(index))
    assert result["weight"] == {"shape": [3], "dtype": "F16"}
    assert "unused" not in result


def test_pytorch_existing_shard_missing_an_indexed_tensor_is_rejected(tmp_path):
    shard = tmp_path / "shard.bin"
    write_pytorch_archive(shard, {"unrelated": {"shape": [1], "dtype": "F16"}})
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    with pytest.raises(ValueError, match="weight"):
        parse_pytorch_index(str(index))


def test_parse_pytorch_header_accepts_an_index_path(tmp_path):
    index, shard_one, shard_two = write_sharded_model(tmp_path)
    result = parse_pytorch_header(str(index))
    assert result["a"] == {"shape": [2], "dtype": "F16"}
    assert result["b"] == {"shape": [3], "dtype": "F32"}
    assert result["__metadata__"] == {
        "missing_shards": 0,
        "total_shards": 2,
        "is_sharded": True,
        "disk_size": shard_one.stat().st_size + shard_two.stat().st_size,
    }


def test_single_shard_next_to_its_index_picks_up_the_index(tmp_path):
    index, shard_one, shard_two = write_sharded_model(tmp_path)
    result = parse_pytorch_header(str(shard_one))
    assert result["a"] == {"shape": [2], "dtype": "F16"}
    assert result["b"] == {"shape": [3], "dtype": "F32"}
    assert result["__metadata__"]["total_shards"] == 2
    assert result["__metadata__"]["is_sharded"] is True


def test_single_shard_without_an_index_reports_only_that_shard(tmp_path):
    shard = tmp_path / "pytorch_model-00001-of-00002.bin"
    write_pytorch_archive(shard, {"a": {"shape": [2], "dtype": "F16"}})
    result = parse_pytorch_header(str(shard))
    assert result == {"a": {"shape": [2], "dtype": "F16"}}


def test_analyze_model_routes_bin_index_to_pytorch(tmp_path):
    index, shard_one, shard_two = write_sharded_model(tmp_path)
    info = cli.analyze_model(str(index), context_override=128)
    assert info["format_name"] == "PyTorch"
    assert info["tensor_count"] == 2
    assert info["disk_size"] == shard_one.stat().st_size + shard_two.stat().st_size


def test_analyze_model_routes_generic_index_by_weight_map(tmp_path):
    shard = tmp_path / "weights-00001-of-00001.pt"
    write_pytorch_archive(shard, {"a": {"shape": [2], "dtype": "F16"}})
    index = tmp_path / "model.index.json"
    index.write_text(json.dumps({"weight_map": {"a": shard.name}}))
    info = cli.analyze_model(str(index), context_override=128)
    assert info["format_name"] == "PyTorch"
    assert info["tensor_count"] == 1


def test_analyze_model_single_bin_shard_picks_up_its_index(tmp_path):
    index, shard_one, shard_two = write_sharded_model(tmp_path)
    info = cli.analyze_model(str(shard_one), context_override=128)
    assert info["format_name"] == "PyTorch"
    assert info["tensor_count"] == 2


def test_analyze_model_keeps_safetensors_index_routing(tmp_path):
    import struct

    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {"a": {"shape": [2], "dtype": "F16", "data_offsets": [0, 4]}}
    payload = json.dumps(header).encode()
    shard.write_bytes(struct.pack("<Q", len(payload)) + payload)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": shard.name}}))
    info = cli.analyze_model(str(index), context_override=128)
    assert info["format_name"] == "SafeTensors"
    assert info["tensor_count"] == 1
