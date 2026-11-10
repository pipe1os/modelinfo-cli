import json
import struct

from modelinfo.parsers.safetensors import parse_safetensors_header


def test_shard_index_selects_only_assigned_tensors(tmp_path):
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'weight': 'part.safetensors'}}))
    header = {'weight': {'shape': [3], 'dtype': 'F16'},
              'unused': {'shape': [1000], 'dtype': 'F32'}}
    data = json.dumps(header).encode()
    (tmp_path / 'part.safetensors').write_bytes(struct.pack('<Q', len(data)) + data)
    result = parse_safetensors_header(str(index))
    assert result['weight'] == header['weight']
    assert 'unused' not in result


def test_existing_shard_missing_an_indexed_tensor_is_rejected(tmp_path):
    import pytest
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'weight': 'part.safetensors'}}))
    data = json.dumps({'unrelated': {'shape': [1], 'dtype': 'F16'}}).encode()
    (tmp_path / 'part.safetensors').write_bytes(struct.pack('<Q', len(data)) + data)
    with pytest.raises(ValueError, match='weight'):
        parse_safetensors_header(str(index))
