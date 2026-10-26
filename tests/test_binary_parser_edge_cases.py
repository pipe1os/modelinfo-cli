"""Edge cases for the local SafeTensors and PyTorch parsers."""

import json
import pickle
import struct
import zipfile

import pytest

from modelinfo.parsers.pytorch import dummy_rebuild_tensor_v2, parse_pytorch_header
from modelinfo.parsers.safetensors import parse_safetensors_header


def write_safetensors(path, header):
    payload = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


@pytest.mark.parametrize("header", [[], "text", 3, None])
def test_safetensors_rejects_non_object_header(tmp_path, header):
    path = tmp_path / "model.safetensors"
    write_safetensors(path, header)
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_safetensors_header(str(path))


def test_safetensors_accepts_empty_header_object(tmp_path):
    path = tmp_path / "empty.safetensors"
    write_safetensors(path, {})
    assert parse_safetensors_header(str(path)) == {}


def test_safetensors_header_length_at_limit_is_not_rejected(tmp_path):
    path = tmp_path / "limit.safetensors"
    path.write_bytes(struct.pack("<Q", 100 * 1024 * 1024))
    # The size check passes; the file is then reported as truncated instead.
    with pytest.raises(EOFError, match="reading JSON header"):
        parse_safetensors_header(str(path))


def test_safetensors_shard_without_index_is_read_directly(tmp_path):
    header = {"w": {"shape": [1], "dtype": "F16", "data_offsets": [0, 2]}}
    path = tmp_path / "model-00001-of-00002.safetensors"
    write_safetensors(path, header)
    assert parse_safetensors_header(str(path)) == header


@pytest.mark.parametrize(
    "shard_name,index_name",
    [
        ("model-00001-of-00002.safetensors", "model.safetensors.index.json"),
        # Names that do not match the numeric pattern use the text before
        # the first dash as the index prefix.
        ("model-a-of-b.safetensors", "model.safetensors.index.json"),
    ],
)
def test_safetensors_shard_path_uses_neighbouring_index(
    tmp_path, shard_name, index_name
):
    first = {"a": {"shape": [2], "dtype": "F16"}}
    second = {"b": {"shape": [3], "dtype": "F32"}}
    write_safetensors(tmp_path / shard_name, first)
    write_safetensors(tmp_path / "other.safetensors", second)
    (tmp_path / index_name).write_text(
        json.dumps({"weight_map": {"a": shard_name, "b": "other.safetensors"}})
    )
    result = parse_safetensors_header(str(tmp_path / shard_name))
    assert result["a"] == first["a"]
    assert result["b"] == second["b"]
    assert result["__metadata__"]["total_shards"] == 2
    assert result["__metadata__"]["missing_shards"] == 0


def test_safetensors_index_rejects_tensor_missing_from_shard(tmp_path):
    shard = tmp_path / "part1.safetensors"
    write_safetensors(shard, {"a": {"shape": [2], "dtype": "F16"}})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"b": shard.name}}))
    with pytest.raises(ValueError, match="missing from shard"):
        parse_safetensors_header(str(index))


def test_safetensors_index_propagates_corrupt_shard(tmp_path):
    shard = tmp_path / "part1.safetensors"
    shard.write_bytes(b"short")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": shard.name}}))
    with pytest.raises(EOFError):
        parse_safetensors_header(str(index))


def test_safetensors_index_without_weight_map_reports_no_shards(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text("{}")
    assert parse_safetensors_header(str(index)) == {
        "__metadata__": {
            "missing_shards": 0,
            "total_shards": 0,
            "is_sharded": True,
            "disk_size": 0,
        }
    }


def test_rebuild_tensor_defaults_to_f32_without_dummy_storage():
    assert dummy_rebuild_tensor_v2(None, 0, (4,), (1,), False, None) == {
        "shape": [4],
        "dtype": "F32",
    }


@pytest.mark.parametrize("data", [[1, 2], "text", None, 7])
def test_pytorch_non_dict_top_level_object_gives_no_tensors(tmp_path, data):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", pickle.dumps(data, protocol=2))
    assert parse_pytorch_header(str(path)) == {}


def test_pytorch_empty_state_dict_gives_no_tensors(tmp_path):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", pickle.dumps({}, protocol=2))
    assert parse_pytorch_header(str(path)) == {}


def test_pytorch_rejects_zip_pickle_with_garbage_bytes(tmp_path):
    path = tmp_path / "garbage.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", b"\xff\xfe not a pickle")
    with pytest.raises(pickle.UnpicklingError):
        parse_pytorch_header(str(path))


def test_pytorch_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.pt"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="not a valid zip archive"):
        parse_pytorch_header(str(path))


def test_pytorch_uses_first_data_pkl_entry(tmp_path):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/data.pkl", pickle.dumps({"x": {"shape": [1]}}, protocol=2))
        archive.writestr("b/data.pkl", pickle.dumps({"y": {"shape": [2]}}, protocol=2))
    assert list(parse_pytorch_header(str(path))) == ["x"]


def test_pytorch_non_tensor_values_get_scalar_placeholder(tmp_path):
    path = tmp_path / "model.pt"
    data = {"w": {"shape": [2], "dtype": "F16"}, "step": 5, "name": "run"}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", pickle.dumps(data, protocol=2))
    result = parse_pytorch_header(str(path))
    assert result["w"] == {"shape": [2], "dtype": "F16"}
    assert result["step"] == {"shape": [], "dtype": "F32"}
    assert result["name"] == {"shape": [], "dtype": "F32"}
