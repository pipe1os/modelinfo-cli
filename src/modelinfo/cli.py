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
        default=0,
        help="Context length for dynamic KV cache footprint calculation.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    file_path = args.file.lower()
    tensors = {}
    config = None
    
    if file_path.endswith(".safetensors") or file_path.endswith(".index.json"):
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
            f"[red]Error: Unsupported file format '{args.file}'. Supported formats are .safetensors, .gguf, .pt[/red]"
        )
        return 1
        
    footprint = calculate_footprint(tensors, context_length=args.context, config=config)
    num_layers = footprint["num_layers"]
    arch_name = identify_architecture_name(tensors, num_layers)
    
    max_context = None
    if config:
        max_context = config.get("max_position_embeddings")
    else:
        metadata = tensors.get("__metadata__", {})
        gen_arch = metadata.get("general.architecture")
        if gen_arch:
            max_context = metadata.get(f"{gen_arch}.context_length")

    disk_size = os.path.getsize(args.file) if os.path.exists(args.file) else 0.0
    tensor_count = len([k for k in tensors.keys() if k != "__metadata__"])
    
    print_model_info(
        format_name=format_name,
        arch_name=arch_name,
        tensor_count=tensor_count,
        footprint=footprint,
        disk_size=disk_size,
        context_length=args.context,
        tensors=tensors,
        max_context=max_context
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
