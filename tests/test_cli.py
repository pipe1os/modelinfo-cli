import pytest
from cli import main

def test_batch_size_flag():
    # test that the --batch-size flag is accepted and passed to analyze_model
    with pytest.raises(SystemExit) as e:
        main(['--batch-size', '2', 'model.safetensors'])
    assert e.value.code == 0