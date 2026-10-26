import pytest

from modelinfo.hardware import _parse_intel_vram


@pytest.mark.parametrize('text', ['-16 GiB', '16 XB', 'unavailable 16 GiB', '1.2.3 GiB', '0 MiB'])
def test_invalid_intel_memory_is_not_reported_as_capacity(text):
    assert _parse_intel_vram(text) is None


def test_intel_terabyte_capacity_converts_to_mib():
    assert _parse_intel_vram('1 TiB') == 1024 * 1024
