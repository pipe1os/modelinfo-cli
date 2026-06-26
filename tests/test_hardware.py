import subprocess

import pytest

from modelinfo import hardware


def completed(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_normalize_gpu_string_removes_vendor_fluff_and_separators():
    assert hardware.normalize_gpu_string("NVIDIA GeForce RTX 4090") == "rtx4090"
    assert (
        hardware.normalize_gpu_string("AMD Radeon RX-7900 XTX Graphics") == "rx7900xtx"
    )
    assert hardware.normalize_gpu_string("Intel Arc A770 Edition") == "a770"


def test_resolve_gpu_matches_known_gpu():
    assert hardware.resolve_gpu("NVIDIA GeForce RTX 4090") == (
        "NVIDIA GeForce RTX 4090",
        24.0,
        1,
    )


def test_resolve_gpu_handles_multi_gpu_string():
    assert hardware.resolve_gpu("2x RTX4090") == ("2x rtx4090", 48.0, 2)


def test_resolve_gpu_accepts_numeric_vram_target():
    assert hardware.resolve_gpu("16") == ("Custom (16.0 GB)", 16.0, 1)
    assert hardware.resolve_gpu("4x 12") == ("Custom (48.0 GB)", 48.0, 4)


def test_resolve_gpu_delegates_auto_detection(monkeypatch):
    monkeypatch.setattr(hardware, "detect_local_gpu", lambda: ("Local GPU", 12.0, 1))

    assert hardware.resolve_gpu("auto") == ("Local GPU", 12.0, 1)


def test_resolve_gpu_rejects_apple_silicon_shortcuts():
    with pytest.raises(ValueError, match="Apple Silicon VRAM varies"):
        hardware.resolve_gpu("m3-max")


def test_resolve_gpu_rejects_unknown_gpu_name():
    with pytest.raises(ValueError, match="Unknown GPU target 'Mystery GPU'"):
        hardware.resolve_gpu("Mystery GPU")


def test_resolve_gpu_suggests_close_matches():
    with pytest.raises(
        ValueError,
        match="Unknown GPU target 'rtx490'\\. Did you mean:.*rtx4090",
    ):
        hardware.resolve_gpu("rtx490")


def test_detect_local_gpu_reads_nvidia_smi(monkeypatch):
    def fake_run(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": True,
            "timeout": 2.0,
        }
        return completed("NVIDIA GeForce RTX 4090, 24576\n")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware.detect_local_gpu() == ("NVIDIA GeForce RTX 4090", 24.0, 1)


def test_detect_local_gpu_sums_multiple_nvidia_gpus(monkeypatch):
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *args, **kwargs: completed(
            "NVIDIA GeForce RTX 4090, 24576\nNVIDIA GeForce RTX 4090, 24576\n"
        ),
    )

    assert hardware.detect_local_gpu() == (
        "Multi-GPU: 2x NVIDIA GeForce RTX 4090",
        48.0,
        2,
    )


def test_detect_local_gpu_falls_back_to_rocm_smi(monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi not installed")
        assert command == ["rocm-smi", "--showmeminfo", "vram"]
        return completed(
            "Total Memory (B): 17179869184\nTotal Memory (B): 17179869184\n"
        )

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware.detect_local_gpu() == ("AMD Multi-GPU (2x)", 32.0, 2)


def test_detect_local_gpu_falls_back_to_xpu_smi(monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] in {"nvidia-smi", "rocm-smi"}:
            raise FileNotFoundError(command[0])
        assert command == ["xpu-smi", "discovery"]
        stdout = (
            "+-----------+------------------------------------------------------+\n"
            "| Device ID | Device Information                                   |\n"
            "+-----------+------------------------------------------------------+\n"
            "| 0         | Device Name: Intel(R) Arc(TM) A770 Graphics          |\n"
            "|           | Vendor Name: Intel(R) Corporation                    |\n"
            "|           | Memory Physical Size: 16384.00 MiB                   |\n"
            "+-----------+------------------------------------------------------+\n"
        )
        return completed(stdout)

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware.detect_local_gpu() == ("Intel(R) Arc(TM) A770 Graphics", 16.0, 1)


def test_detect_local_gpu_sums_multiple_intel_gpus(monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] in {"nvidia-smi", "rocm-smi"}:
            raise FileNotFoundError(command[0])
        assert command == ["xpu-smi", "discovery"]
        stdout = (
            "+-----------+------------------------------------------------------+\n"
            "| Device ID | Device Information                                   |\n"
            "+-----------+------------------------------------------------------+\n"
            "| 0         | Device Name: Intel(R) Data Center GPU Flex 170       |\n"
            "|           | Memory Physical Size: 16384.00 MiB                   |\n"
            "+-----------+------------------------------------------------------+\n"
            "| 1         | Device Name: Intel(R) Data Center GPU Flex 170       |\n"
            "|           | Memory Physical Size: 16384.00 MiB                   |\n"
            "+-----------+------------------------------------------------------+\n"
        )
        return completed(stdout)

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware.detect_local_gpu() == (
        "Intel Multi-GPU (2x Intel(R) Data Center GPU Flex 170)",
        32.0,
        2,
    )


def test_detect_local_gpu_intel_unit_conversions(monkeypatch):
    test_cases = [
        ("16.00 GiB", 16.0),
        ("16.00 GB", 16.0),
        ("16777216.00 KiB", 16.0),
        ("17179869184.00 B", 16.0),
        ("16384.00 MiB", 16.0),
        ("16384.00 MB", 16.0),
        ("16384.00", 16.0),  # Default MiB unit
    ]
    for size_str, expected_vram in test_cases:
        def fake_run(command, **kwargs):
            if command[0] in {"nvidia-smi", "rocm-smi"}:
                raise FileNotFoundError(command[0])
            assert command == ["xpu-smi", "discovery"]
            stdout = (
                "+-----------+------------------------------------------------------+\n"
                "| Device ID | Device Information                                   |\n"
                "+-----------+------------------------------------------------------+\n"
                "| 0         | Device Name: Intel(R) Arc(TM) A770 Graphics          |\n"
                f"|           | Memory Physical Size: {size_str}                     |\n"
                "+-----------+------------------------------------------------------+\n"
            )
            return completed(stdout)

        monkeypatch.setattr(hardware.subprocess, "run", fake_run)
        assert hardware.detect_local_gpu() == ("Intel(R) Arc(TM) A770 Graphics", expected_vram, 1)


def test_detect_local_gpu_falls_back_on_malformed_xpu_smi(monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] in {"nvidia-smi", "rocm-smi"}:
            raise FileNotFoundError(command[0])
        if command[0] == "xpu-smi":
            # Returns device name but no parseable memory size
            stdout = (
                "+-----------+------------------------------------------------------+\n"
                "| Device ID | Device Information                                   |\n"
                "+-----------+------------------------------------------------------+\n"
                "| 0         | Device Name: Intel(R) Arc(TM) A770 Graphics          |\n"
                "|           | Vendor Name: Intel(R) Corporation                    |\n"
                "|           | Memory Physical Size: N/A                            |\n"
                "+-----------+------------------------------------------------------+\n"
            )
            return completed(stdout)
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    # Since xpu-smi didn't return valid memory, detect_local_gpu should fall back to default/next
    assert hardware.detect_local_gpu() == ("Unknown", 8.0, 1)


def test_detect_local_gpu_falls_back_to_apple_unified_memory(monkeypatch):
    def fake_run(command, **kwargs):
        if command[0] in {"nvidia-smi", "rocm-smi", "xpu-smi"}:
            raise FileNotFoundError(command[0])
        assert command == ["sysctl", "hw.memsize"]
        return completed("hw.memsize: 17179869184\n")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware.detect_local_gpu() == ("Apple Silicon (Unified Memory)", 12.0, 1)


def test_detect_local_gpu_returns_default_when_detection_fails(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware.detect_local_gpu() == ("Unknown", 8.0, 1)
