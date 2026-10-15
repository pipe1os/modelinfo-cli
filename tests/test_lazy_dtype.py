import pytest

from modelinfo.calculator import calculate_footprint


@pytest.mark.parametrize('key,dtype,expected,width', [
    ('torch_dtype', 'float32', 'F32', 4),
    ('dtype', 'float64', 'F64', 8),
    ('torch_dtype', 'float16', 'F16', 2),
])
def test_lazy_parameter_estimate_respects_declared_dtype(key, dtype, expected, width):
    result = calculate_footprint({'__metadata__': {'lazy_fetch': True, 'total_size': 1024}}, config={key: dtype})
    assert result['primary_dtype'] == expected
    assert result['total_params'] == 1024 // width
    assert result['base_memory_bytes'] == 1024


def test_lazy_dtype_falls_back_to_top_level_when_text_config_omits_it():
    config = {'torch_dtype': 'float32', 'text_config': {'num_hidden_layers': 2}}
    tensors = {'__metadata__': {'lazy_fetch': True, 'total_size': 1024}}
    result = calculate_footprint(tensors, config=config)
    assert result['primary_dtype'] == 'F32'
    assert result['total_params'] == 256
