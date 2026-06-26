import re
import subprocess
from typing import Tuple

KNOWN_GPUS = {
    # --- NVIDIA Consumer (RTX 50/40/30/20/10 Series & Titans) ---
    "rtx5090": 32.0,
    "rtx5080": 16.0,
    "rtx5070ti": 16.0,
    "rtx5070": 12.0,
    "rtx5060ti16gb": 16.0,
    "rtx5060ti": 8.0,
    "rtx5060": 8.0,
    "rtx4090": 24.0,
    "rtx4080super": 16.0,
    "rtx4080": 16.0,
    "rtx4070tisuper": 16.0,
    "rtx4070ti": 12.0,
    "rtx4070super": 12.0,
    "rtx4070": 12.0,
    "rtx4060ti16gb": 16.0,
    "rtx4060ti": 8.0,
    "rtx4060": 8.0,
    "rtx4050": 6.0,
    "rtx3090ti": 24.0,
    "rtx3090": 24.0,
    "rtx3080ti": 12.0,
    "rtx308012gb": 12.0,
    "rtx3080": 10.0,
    "rtx3070ti": 8.0,
    "rtx3070": 8.0,
    "rtx3060ti": 8.0,
    "rtx306012gb": 12.0,
    "rtx3060": 8.0,
    "rtx3050ti": 4.0,
    "rtx3050": 8.0,
    "rtx2080ti": 11.0,
    "rtx2080super": 8.0,
    "rtx2080": 8.0,
    "rtx2070super": 8.0,
    "rtx2070": 8.0,
    "rtx2060super": 8.0,
    "rtx206012gb": 12.0,
    "rtx2060": 6.0,
    "gtx1660super": 6.0,
    "gtx1660ti": 6.0,
    "gtx1660": 6.0,
    "gtx1650super": 4.0,
    "gtx1650": 4.0,
    "gtx1080ti": 11.0,
    "gtx1080": 8.0,
    "gtx1070ti": 8.0,
    "gtx1070": 8.0,
    "gtx1060": 6.0,
    "titanrtx": 24.0,
    "titanv": 12.0,
    "titanxp": 12.0,
    "titanxpascal": 12.0,
    "titanx": 12.0,
    # --- NVIDIA Data Center / Workstation ---
    "b200": 192.0,
    "b100": 192.0,
    "h200": 141.0,
    "h100nvl": 188.0,
    "h100": 80.0,
    "a10080gb": 80.0,
    "a100": 40.0,
    "a40": 48.0,
    "a30": 24.0,
    "a10g": 24.0,
    "a10": 24.0,
    "l40s": 48.0,
    "l40": 48.0,
    "l4": 24.0,
    "v10032gb": 32.0,
    "v100": 16.0,
    "p100": 16.0,
    "p40": 24.0,
    "m40": 24.0,
    "rtx6000ada": 48.0,
    "rtx5000ada": 32.0,
    "rtx4500ada": 24.0,
    "rtx4000ada": 20.0,
    "rtxa6000": 48.0,
    "rtxa5500": 24.0,
    "rtxa5000": 24.0,
    "rtxa4500": 20.0,
    "rtxa4000": 16.0,
    "quadrortx8000": 48.0,
    "quadrortx6000": 24.0,
    # --- AMD Consumer (RX 9000/7000/6000 Series) ---
    "rx9070xt": 16.0,
    "rx9070": 16.0,
    "rx9070gre": 12.0,
    "rx9060xt": 16.0,
    "rx9060": 8.0,
    "rx7900xtx": 24.0,
    "rx7900xt": 20.0,
    "rx7900gre": 16.0,
    "rx7800xt": 16.0,
    "rx7700xt": 12.0,
    "rx7600xt": 16.0,
    "rx7600": 8.0,
    "rx6950xt": 16.0,
    "rx6900xt": 16.0,
    "rx6800xt": 16.0,
    "rx6800": 16.0,
    "rx6750xt": 12.0,
    "rx6700xt": 12.0,
    "rx6700": 10.0,
    "rx6650xt": 8.0,
    "rx6600xt": 8.0,
    "rx6600": 8.0,
    "rx580": 8.0,
    "rx570": 4.0,
    # --- AMD Data Center / Pro ---
    "mi300x": 192.0,
    "mi250x": 128.0,
    "mi210": 64.0,
    "prow7900": 48.0,
    "prow7800": 32.0,
    "prow6800": 32.0,
    # --- Intel Consumer & Accelerators ---
    "arcb580": 12.0,
    "b580": 12.0,
    "arcb570": 10.0,
    "b570": 10.0,
    "arca770": 16.0,
    "a770": 16.0,
    "arca750": 8.0,
    "a750": 8.0,
    "gaudi3": 128.0,
    "gaudi2": 96.0,
}


def normalize_gpu_string(name: str) -> str:
    """Strips vendor fluff, spaces, and hyphens to map correctly to KNOWN_GPUS."""
    name = name.lower()

    # Remove common vendor/marketing fluff that disrupts core identifiers
    fluff_words = [
        "nvidia",
        "geforce",
        "amd",
        "radeon",
        "intel",
        "arc",
        "generation",
        "edition",
        "graphics",
        "accelerator",
    ]
    for word in fluff_words:
        name = name.replace(word, "")

    return re.sub(r"[\s\-]", "", name)


def detect_local_gpu() -> Tuple[str, float, int]:
    # 1. NVIDIA
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ]
        if lines:
            total_mb = 0
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 2:
                    total_mb += int(parts[1].strip())

            gpu_count = len(lines)
            first_name = lines[0].split(",")[0].strip()
            display_name = (
                f"Multi-GPU: {gpu_count}x {first_name}" if gpu_count > 1 else first_name
            )
            return display_name, total_mb / 1024.0, gpu_count
    except Exception:
        pass

    # 2. AMD (ROCm)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [
            line
            for line in result.stdout.strip().split("\n")
            if "Total Memory (B):" in line
        ]
        if lines:
            total_bytes = 0
            gpu_count = len(lines)
            for line in lines:
                parts = line.split(":")
                if len(parts) >= 2:
                    total_bytes += int(parts[1].strip())
            display_name = (
                f"AMD Multi-GPU ({gpu_count}x)" if gpu_count > 1 else "AMD GPU"
            )
            return display_name, total_bytes / (1024.0**3), gpu_count
    except Exception:
        pass

    # 3. Apple Silicon
    try:
        result = subprocess.run(
            ["sysctl", "hw.memsize"], capture_output=True, text=True, check=True
        )
        total_bytes = int(result.stdout.strip().split()[1])
        # Apply 75% operational heuristic for Apple Silicon wire limits
        vram_gb = (total_bytes / (1024.0**3)) * 0.75
        return "Apple Silicon (Unified Memory)", vram_gb, 1
    except Exception:
        pass

    return "Unknown", 8.0, 1


def resolve_gpu(target: str) -> Tuple[str, float, int]:
    if target.lower() == "auto":
        return detect_local_gpu()

    # Apple Silicon routing trap
    lower_target = target.lower()
    if lower_target in ["m1", "m2", "m3", "m4", "apple", "mac"] or re.match(
        r"^m[1-4](-?(pro|max|ultra))?$", lower_target
    ):
        raise ValueError(
            "Apple Silicon VRAM varies by machine configuration. Please use '--gpu auto' to calculate your specific Unified Memory limits."
        )

    # Parse potential multi-GPU format e.g., "2x RTX4090"
    gpu_count = 1
    match = re.match(r"^(\d+)x\s*(.+)$", lower_target)
    if match:
        gpu_count = int(match.group(1))
        target_name = match.group(2)
    else:
        target_name = target

    normalized = normalize_gpu_string(target_name)

    if normalized in KNOWN_GPUS:
        vram_gb = KNOWN_GPUS[normalized] * gpu_count
        display_name = f"{gpu_count}x {target_name}" if gpu_count > 1 else target_name
        return display_name, vram_gb, gpu_count

    # If the user passed a pure number, assume GB
    try:
        vram_gb = float(normalized) * gpu_count
        display_name = f"Custom ({vram_gb} GB)"
        return display_name, vram_gb, gpu_count
    except ValueError:
        pass

    import difflib

    matches = difflib.get_close_matches(normalized, KNOWN_GPUS.keys(), n=3, cutoff=0.6)
    if matches:
        suggestions = ", ".join(matches)
        raise ValueError(
            f"Unknown GPU target '{target}'. Did you mean: {suggestions}? "
            f"Use '--gpu auto' to detect automatically, or provide a known name (e.g., 'RTX4090') or a numeric GB value."
        )

    raise ValueError(
        f"Unknown GPU target '{target}'. Use '--gpu auto' to detect automatically, or provide a known name (e.g., 'RTX4090') or a numeric GB value."
    )
