import re
import subprocess
from typing import Tuple, Optional

KNOWN_GPUS = {
    # ... (unchanged)
}

def normalize_gpu_string(name: str) -> str:
    """Strips vendor fluff, spaces, and hyphens to map correctly to KNOWN_GPUS."""
    name = name.lower()
    
    # Remove common vendor/marketing fluff that disrupts core identifiers
    fluff_words = ["nvidia", "geforce", "amd", "radeon", "intel", "arc", "generation", "edition", "graphics", "accelerator"]
    for word in fluff_words:
        name = name.replace(word, "")
            
    return re.sub(r'[\s\-]', '', name)

def resolve_gpu(name: str) -> Optional[Tuple[str, float]]:
    """Resolves a GPU name to its corresponding memory size."""
    normalized_name = normalize_gpu_string(name)
    return KNOWN_GPUS.get(normalized_name, None)

def detect_local_gpu() -> Optional[Tuple[str, float, int]]:
    # 1. NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        if lines:
            total_mb = 0
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    total_mb += int(parts[1].strip())
            
            gpu_count = len(lines)
            first_name = lines[0].split(',')[0].strip()
            if gpu_count > 1:
                display_name = f"Multi-GPU: {gpu_count}x {first_name}" 
            else:
                display_name = first_name

            return display_name, total_mb, gpu_count
    except subprocess.CalledProcessError:
        pass

    return None