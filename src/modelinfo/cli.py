import argparse
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
    
    if file_path.endswith(".safetensors") or file_path.endswith(".index.json"):
        tensors = parse_safetensors_header(args.file)
        format_name = "SafeTensors"
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
        
    footprint = calculate_footprint(tensors, context_length=args.context)
    num_layers = footprint["num_layers"]
    arch_name = identify_architecture_name(tensors, num_layers)
    
    disk_size = os.path.getsize(args.file) if os.path.exists(args.file) else 0.0
    tensor_count = len([k for k in tensors.keys() if k != "__metadata__"])
    
    print_model_info(
        format_name=format_name,
        arch_name=arch_name,
        tensor_count=tensor_count,
        footprint=footprint,
        disk_size=disk_size,
        context_length=args.context,
        tensors=tensors
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
