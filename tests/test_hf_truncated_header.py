import struct
from unittest.mock import patch

import pytest

from modelinfo.parsers.huggingface import _fetch_safetensors_header


@pytest.mark.parametrize('declared_size', [64, 600000])
def test_remote_header_rejects_truncation_even_when_json_is_complete(declared_size):
    first_chunk = struct.pack('<Q', declared_size) + b'{}'
    with (
        patch('modelinfo.parsers.huggingface._make_request', side_effect=[first_chunk, b'{}']),
        pytest.raises(ValueError, match='truncated'),
    ):
        _fetch_safetensors_header('org/model', 'model.safetensors')
