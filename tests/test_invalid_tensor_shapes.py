import pytest

from modelinfo.calculator import calculate_footprint


@pytest.mark.parametrize('shape', [[-1, 4], [True, 4], [1.5, 4], ['3', 4], '34'])
def test_invalid_tensor_dimensions_are_rejected(shape):
    with pytest.raises(ValueError, match='shape'):
        calculate_footprint({'weight': {'shape': shape, 'dtype': 'F16'}})


def test_zero_length_dimensions_and_scalar_shapes_remain_valid():
    result = calculate_footprint({'empty': {'shape': [0, 4], 'dtype': 'F16'},
                                  'scalar': {'shape': [], 'dtype': 'F32'}})
    assert result['total_params'] == 1
    assert result['base_memory_bytes'] == 4
