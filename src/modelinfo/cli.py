import argparse
import json
import os
import sys
from typing import Sequence

from modelinfo.architecture import identify_architecture_name
from modelinfo.calculator import calculate_footprint
from modelinfo.parsers.gguf import parse_gguf_header
from modelinfo.parsers.pytorch import parse_pytorch_header
from modelinfo.parsers.safetensors import parse_safetensors_header
from modelinfo.ui import console, print_model_info, print_compare_info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="modelinfo",
        description="High-performance CLI utility to inspect ML model checkpoints and calculate VRAM requirements.",
    )
    
    parser.add_argument(
        "file",
        type=str,
        nargs="+",
        help="Path to the model checkpoint file(s) or Hugging Face repository IDs.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Context length for dynamic KV cache footprint calculation.",
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=8.0,
        help="Maximum VRAM in GB for color-coding thresholds.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Target GPU hardware (e.g. 'RTX4090' or 'auto') to check if the model fits.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        help="Deep dive: Fetch all remote tensor shards to display the exact tensor size breakdown.",
    )

    return parser.parse_args(argv)


def analyze_model(file_path: str, context_override: int | None, gpu_count: int = 1, fetch_tensors: bool = False) -> dict:
    tensors = {}
    config = None
    disk_size = 0.0
    
    file_path_lower = file_path.lower()
    
    if not os.path.exists(file_path) and not file_path_lower.endswith((".safetensors", ".gguf", ".pt", ".bin", ".index.json")):
        from modelinfo.parsers.huggingface import fetch_huggingface_repo
        tensors, config, format_name, disk_size = fetch_huggingface_repo(file_path, fetch_tensors=fetch_tensors)
    elif file_path_lower.endswith(".safetensors") or file_path_lower.endswith(".index.json"):
        tensors = parse_safetensors_header(file_path)
        format_name = "SafeTensors"
        
        config_path = os.path.join(os.path.dirname(file_path), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
                
    elif file_path_lower.endswith(".gguf"):
        tensors = parse_gguf_header(file_path)
        format_name = "GGUF"
    elif file_path_lower.endswith(".pt") or file_path_lower.endswith(".bin"):
        tensors = parse_pytorch_header(file_path)
        format_name = "PyTorch"
    else:
        raise ValueError(f"File '{file_path}' not found locally and does not appear to be a Hugging Face repository ID.")
        
    max_context = None
    if config:
        max_context = config.get("max_position_embeddings")
    elif format_name == "GGUF":
        metadata = tensors.get("__metadata__", {})
        gen_arch = metadata.get("general.architecture")
        if gen_arch:
            max_context = metadata.get(f"{gen_arch}.context_length")
            
    is_default_context = False
    context_length = context_override
    if context_length is None:
        context_length = min(8192, max_context) if max_context else 8192
        is_default_context = True

    footprint = calculate_footprint(tensors, context_length=context_length, config=config, gpu_count=gpu_count)
    num_layers = footprint["num_layers"]
    arch_name = identify_architecture_name(tensors, num_layers, config)

    if format_name != "SafeTensors" or os.path.exists(file_path):
        disk_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0.0
        
    tensor_count = len([k for k in tensors.keys() if k != "__metadata__"])
    
    return {
        "format_name": format_name,
        "arch_name": arch_name,
        "tensor_count": tensor_count,
        "footprint": footprint,
        "disk_size": disk_size,
        "context_length": context_length,
        "is_default_context": is_default_context,
        "tensors": tensors,
        "max_context": max_context,
        "is_lazy": tensors.get("__metadata__", {}).get("lazy_fetch", False)
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    gpu_name_display = None
    gpu_count = 1
    if args.gpu:
        from modelinfo.hardware import resolve_gpu
        try:
            gpu_name_display, args.max_vram, gpu_count = resolve_gpu(args.gpu)
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            return 1

    if len(args.file) > 1:
        models = []
        for model_path in args.file:
            try:
                info = analyze_model(model_path, args.context, gpu_count, fetch_tensors=args.tensors)
                models.append((model_path.split("/")[-1], info))
            except Exception as e:
                console.print(f"[red]Error analyzing model '{model_path}': {e}[/red]")
                return 1
            
        print_compare_info(models, args.max_vram, gpu_name=gpu_name_display)
        return 0
        
    file_path = args.file[0]
    
    try:
        info = analyze_model(file_path, args.context, gpu_count, fetch_tensors=args.tensors)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    print_model_info(**info, max_vram_gb=args.max_vram, gpu_name=gpu_name_display)
    return 0


if __name__ == "__main__":
    sys.exit(main())
