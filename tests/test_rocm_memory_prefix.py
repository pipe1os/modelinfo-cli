import subprocess

from modelinfo import hardware


def test_rocm_memory_rows_with_device_prefix(monkeypatch):
    output = ('GPU[0] : VRAM Total Memory (B): 17179869184\n'
              'GPU[0] : VRAM Total Used Memory (B): 1024\n'
              'GPU[1] : VRAM Total Memory (B): 8589934592\n')
    monkeypatch.setattr(hardware.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 0, output))
    assert hardware._detect_amd_gpu() == ('AMD Multi-GPU (2x)', 24.0, 2)
