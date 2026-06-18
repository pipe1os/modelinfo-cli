import pytest

from unittest.mock import patch, MagicMock
from modelinfo.cli import main
from modelinfo.parsers.huggingface import fetch_huggingface_repo

def test_cli_timeout_argument():
    with patch('modelinfo.parsers.huggingface.fetch_huggingface_repo') as mock_fetch:

        mock_fetch.return_value = ({}, {}, "SafeTensors", 0.0)

        try:
            result = main(["--timeout", "20.0", "gpt2"])
            assert result == 0
        except SystemExit as e:
            pytest.fail(f"CLI failed with SystemExit: {e}. --timeout argument is likely not implemented.")

def test_fetch_huggingface_repo_timeout_parameter():
    with patch('urllib.request.urlopen') as mock_urlopen:
        import struct

        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response

        header_json = b'{"__metadata__": {}}'
        header_size = len(header_json)
        valid_header = struct.pack("<Q", header_size) + header_json
        
        mock_response.read.side_effect = [
            b'{"siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}]}',
            b'{"max_position_embeddings": 1024}',
            valid_header,
        ]

        mock_response.headers = {"Content-Length": "1000"}
        
        mock_urlopen.return_value = mock_response

        try:
            fetch_huggingface_repo("gpt2", timeout=20.0)
        except TypeError as e:
            pytest.fail(f"fetch_huggingface_repo failed with TypeError: {e}. timeout parameter is likely not implemented.")

        args, kwargs = mock_urlopen.call_args
        assert kwargs.get('timeout') == 20.0, f"Expected timeout=20.0, but got {kwargs.get('timeout')}"
