import io
from unittest.mock import patch

import pytest

from modelinfo.parsers.huggingface import _make_request


@pytest.mark.parametrize('content_range', ['bytes 0-7/16', 'bytes 9-16/32', 'invalid'])
def test_partial_response_rejects_wrong_range(content_range):
    response = io.BytesIO(b'bad-data')
    response.status = 206
    response.headers = {'Content-Range': content_range}
    with (
        patch('modelinfo.parsers.huggingface._get_hf_token', return_value=None),
        patch('modelinfo.parsers.huggingface.urllib.request.urlopen', return_value=response),
        pytest.raises(ValueError, match='range'),
    ):
        _make_request('https://example.test/file', {'Range': 'bytes=8-15'}, limit=8)
