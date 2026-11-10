import platform
import subprocess

from modelinfo import hardware


def test_intel_mac_memory_is_not_apple_silicon_vram(monkeypatch):
    def run(command, **kwargs):
        text = '0\n' if command == ['sysctl', '-n', 'hw.optional.arm64'] else 'hw.memsize: 17179869184\n'
        return subprocess.CompletedProcess(command, 0, text)
    monkeypatch.setattr(hardware.subprocess, 'run', run)
    assert hardware._detect_apple_gpu() is None


def test_apple_silicon_is_detected_under_translated_process(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    def run(command, **kwargs):
        text = '1\n' if command == ['sysctl', '-n', 'hw.optional.arm64'] else 'hw.memsize: 17179869184\n'
        return subprocess.CompletedProcess(command, 0, text)
    monkeypatch.setattr(hardware.subprocess, 'run', run)
    assert hardware._detect_apple_gpu() == ('Apple Silicon (Unified Memory)', 12.0, 1)
