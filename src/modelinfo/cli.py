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
from modelinfo.ui import console, print_model_info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="modelinfo",
        description="High-performance CLI utility to inspect ML model checkpoints and calculate VRAM requirements.",
    )
    
    parser.add_argument(
        "file",
        type=str,
        help="Path to the model checkpoint file (.safetensors, .gguf, .pt)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Context length for dynamic KV cache footprint calculation.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    file_path = args.file.lower()
    tensors = {}
    config = None
    
    if not os.path.exists(file_path) and not file_path.endswith((".safetensors", ".gguf", ".pt", ".bin", ".index.json")):
        from modelinfo.parsers.huggingface import fetch_huggingface_repo
        try:
            tensors, config, format_name, disk_size = fetch_huggingface_repo(args.file)
        except Exception as e:
            console.print(f"[red]Error fetching from Hugging Face: {e}[/red]")
            return 1
    elif file_path.endswith(".safetensors") or file_path.endswith(".index.json"):
        tensors = parse_safetensors_header(args.file)
        format_name = "SafeTensors"
        
        # Read config.json to maintain pure math engines
        config_path = os.path.join(os.path.dirname(args.file), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
                
    elif file_path.endswith(".gguf"):
        tensors = parse_gguf_header(args.file)
        format_name = "GGUF"
    elif file_path.endswith(".pt") or file_path.endswith(".bin"):
        tensors = parse_pytorch_header(args.file)
        format_name = "PyTorch"
    else:
        console.print(
            f"[red]Error: File '{args.file}' not found locally and does not appear to be a Hugging Face repository ID.[/red]"
        )
        return 1
        
    max_context = None
    if config:
        max_context = config.get("max_position_embeddings")
    elif format_name == "GGUF":
        metadata = tensors.get("__metadata__", {})
        gen_arch = metadata.get("general.architecture")
        if gen_arch:
            max_context = metadata.get(f"{gen_arch}.context_length")
            
    # Determine the actual context length to use for calculation
    is_default_context = False
    context_length = args.context
    if context_length is None:
        context_length = max_context if max_context else 2048
        is_default_context = True

    footprint = calculate_footprint(tensors, context_length=context_length, config=config)
    num_layers = footprint["num_layers"]
    arch_name = identify_architecture_name(tensors, num_layers, config)

    if format_name != "SafeTensors" or os.path.exists(args.file):
        disk_size = os.path.getsize(args.file) if os.path.exists(args.file) else 0.0
        
    tensor_count = len([k for k in tensors.keys() if k != "__metadata__"])
    
    print_model_info(
        format_name=format_name,
        arch_name=arch_name,
        tensor_count=tensor_count,
        footprint=footprint,
        disk_size=disk_size,
        context_length=context_length,
        is_default_context=is_default_context,
        tensors=tensors,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
