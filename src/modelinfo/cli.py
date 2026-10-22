import argparse
import functools
import json
import math
import os
import pickle  # nosec B403 - only used to catch UnpicklingError, nothing is unpickled here
import struct
import sys
from typing import Callable, Sequence
from rich.markup import escape
from modelinfo.architecture import identify_architecture_name
from modelinfo.calculator import calculate_footprint
from modelinfo.parsers.gguf import parse_gguf_header
from modelinfo.parsers.pytorch import parse_pytorch_header, parse_pytorch_index
from modelinfo.parsers.safetensors import parse_safetensors_header
from modelinfo.ui import (
    error_console,
    fit_status,
    gguf_variant_totals,
    print_compare_info,
    print_model_info,
)


class VersionAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help="show program's version number and exit"):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        from importlib.metadata import PackageNotFoundError, version
        from modelinfo import __version__

        try:
            ver = version("modelinfo-cli")
        except PackageNotFoundError:
            ver = __version__

        print(f"{parser.prog} {ver}")
        parser.exit()


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("batch size must be at least 1")
    return ivalue


def _positive_float(value: str) -> float:
    fvalue = float(value)
    if not math.isfinite(fvalue) or fvalue <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite number greater than 0")
    return fvalue


def _positive_context(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("context must be greater than 0")
    return result


def _positive_vram(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("VRAM must be a finite number greater than 0")
    return result


def _gpu_utilization(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise argparse.ArgumentTypeError("GPU utilization must be greater than 0 and at most 1")
    return result


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
        type=_positive_context,
        default=None,
        help="Context length for dynamic KV cache footprint calculation.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Batch size for dynamic KV cache footprint calculation.",
    )
    parser.add_argument(
        "--max-vram",
        type=_positive_vram,
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
        "--json",
        action="store_true",
        help="Print analysis as JSON instead of terminal tables.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        help="Deep dive: Fetch all remote tensor shards to display the exact tensor size breakdown.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=10.0,
        help="Network timeout in seconds for remote Hugging Face fetches.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=["nvlink", "pcie4", "pcie3"],
        default="pcie4",
        help="Interconnect topology to calculate distributed communication overhead.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["tp", "pp"],
        default="tp",
        help="Distributed parallelism strategy (Tensor vs Pipeline).",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Enable vLLM Capacity Simulation: Calculate max context tokens using PagedAttention allocation.",
    )
    parser.add_argument(
        "--gpu-util",
        type=_gpu_utilization,
        default=0.9,
        help="vLLM gpu_memory_utilization ratio (default 0.9). Reserves 10 percent for PyTorch context.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action=VersionAction,
    )

    return parser.parse_args(argv)


def _is_pytorch_index(file_path: str) -> bool:
    """Return True when an ``*.index.json`` file describes PyTorch shards.

    The canonical ``pytorch_model.bin.index.json`` name (and any
    ``*.pt.index.json``) is PyTorch by name; any other index is PyTorch
    only when every shard its ``weight_map`` points to is a ``.bin``/``.pt``
    file, so SafeTensors indexes keep their existing routing.
    """
    file_path_lower = file_path.lower()
    if file_path_lower.endswith((".bin.index.json", ".pt.index.json")):
        return True
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except (OSError, ValueError):
        return False
    if not isinstance(index_data, dict):
        return False
    weight_map = index_data.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shard_names = [str(name).lower() for name in weight_map.values()]
    return all(name.endswith((".bin", ".pt")) for name in shard_names)


def analyze_model(
    file_path: str, 
    context_override: int | None, 
    gpu_count: int = 1,
    batch_size: int = 1,
    fetch_tensors: bool = False,
    timeout: float = 10.0,
    topology: str = "pcie4",
    strategy: str = "tp",
    is_vllm: bool = False,
    gpu_vram_gb: float = 0.0,
    gpu_util: float = 0.9
) -> dict:
    tensors = {}
    config = None
    disk_size = 0.0
    
    file_path_lower = file_path.lower()
    
    is_remote = False
    if not os.path.exists(file_path):
        is_local_path = (
            file_path.startswith((".", "/", "~"))
            or os.path.isabs(file_path)
        )
        if not is_local_path:
            # Treat as remote only if it contains a slash and does not end with a model extension
            explicit_hub_gguf = (
                len(file_path.split("/")) >= 3 and file_path_lower.endswith(".gguf")
            )
            if explicit_hub_gguf or (
                "/" in file_path and not file_path_lower.endswith((".safetensors", ".gguf", ".pt", ".bin", ".index.json"))
            ):
                is_remote = True

    if is_remote:
        from modelinfo.parsers.huggingface import fetch_huggingface_repo
        tensors, config, format_name, disk_size = fetch_huggingface_repo(
            file_path, fetch_tensors=fetch_tensors, timeout=timeout
        )
    elif file_path_lower.endswith(".safetensors") or file_path_lower.endswith(".index.json"):
        if file_path_lower.endswith(".index.json") and _is_pytorch_index(file_path):
            tensors = parse_pytorch_index(file_path)
            format_name = "PyTorch"
        else:
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
    elif os.path.isdir(file_path):
        raise IsADirectoryError(f"'{file_path}' is a directory. Please provide the path to a specific weights file (e.g. .safetensors, .gguf, .pt) inside the directory.")
    else:
        raise ValueError(f"File '{file_path}' not found locally and does not appear to be a Hugging Face repository ID.")
        
    max_context = None
    if config:
        max_context = config.get("max_position_embeddings")
    elif format_name in ("GGUF", "GGUF_group"):
        metadata = tensors.get("__metadata__", {})
        gen_arch = metadata.get("general.architecture")
        if gen_arch:
            max_context = metadata.get(f"{gen_arch}.context_length")
            
    is_default_context = False
    context_length = context_override
    if context_length is None:
        context_length = min(8192, max_context) if max_context else 8192
        is_default_context = True

    footprint = calculate_footprint(
        tensors, 
        context_length=context_length,
        batch_size=batch_size,
        config=config,
        gpu_count=gpu_count,
        topology=topology,
        strategy=strategy,
        is_vllm=is_vllm,
        gpu_vram_bytes=gpu_vram_gb * 1024**3 if gpu_vram_gb else 0.0,
        gpu_util=gpu_util
    )
    num_layers = footprint["num_layers"]
    arch_name = identify_architecture_name(tensors, num_layers, config)

    if not is_remote:
        metadata = tensors.get("__metadata__", {})
        if metadata.get("is_sharded") and "disk_size" in metadata:
            disk_size = metadata["disk_size"]
        elif os.path.exists(file_path):
            disk_size = os.path.getsize(file_path)
        
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
        "is_lazy": tensors.get("__metadata__", {}).get("lazy_fetch", False),
        "gpu_count": gpu_count,
        "topology": topology,
        "strategy": strategy,
        "is_vllm": is_vllm,
        "gpu_vram_gb": gpu_vram_gb,
        "gpu_util": gpu_util
    }


def _hardware_fit(info: dict, metadata: dict, max_vram_gb: float) -> str:
    """Return the same fit verdict the terminal report shows for a single model."""
    footprint = info["footprint"]
    if info["is_vllm"]:
        paged_kv_pool = footprint.get("vllm_metrics", {}).get("paged_kv_pool", 0.0)
        return "yes" if paged_kv_pool > 0 else "no"
    if metadata.get("missing_shards", 0) > 0 or footprint["total_memory_bytes"] == 0:
        return "unknown"
    return fit_status(footprint["total_memory_bytes"], max_vram_gb, info["gpu_util"])


def _json_metadata(metadata: dict) -> dict:
    # GGUF tokenizer arrays (vocabulary, merges, scores) run to many megabytes.
    return {
        key: value
        for key, value in metadata.items()
        if not (key.startswith("tokenizer.") and isinstance(value, list))
    }


def _json_variants(
    info: dict, metadata: dict, gpu_name: str | None, max_vram_gb: float
) -> list:
    variants = gguf_variant_totals(
        metadata.get("gguf_variants", []), info["footprint"], info["gpu_count"]
    )
    if gpu_name is not None:
        for variant in variants:
            variant["fits"] = fit_status(
                variant["total_vram_bytes"], max_vram_gb, info["gpu_util"]
            )
    return variants


def _json_report(
    info: dict, gpu_name: str | None, max_vram_gb: float, include_tensors: bool
) -> dict:
    """Build the --json document from an analysis without the per-tensor table."""
    report = dict(info)
    tensors = dict(report.pop("tensors"))
    metadata = tensors.pop("__metadata__", {})
    report["metadata"] = _json_metadata(metadata)
    if include_tensors:
        report["tensors"] = tensors

    if info["format_name"] == "GGUF_group":
        report["variants"] = _json_variants(info, metadata, gpu_name, max_vram_gb)
    elif gpu_name is not None:
        report["hardware_fit"] = _hardware_fit(info, metadata, max_vram_gb)
    if gpu_name is not None:
        report["gpu_name"] = gpu_name
    return report


def _report_errors(func: Callable[..., int]) -> Callable[..., int]:
    """Show the actionable messages parsers and the Hub client raise without a traceback."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> int:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            return 130
        except (OSError, ValueError, EOFError, struct.error, pickle.UnpicklingError) as exc:
            error_console.print(f"[red]Error:[/red] {escape(str(exc))}")
            return 1

    return wrapper


@_report_errors
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Strip trailing slashes from paths/repos to prevent empty basenames and routing issues
    if args.file:
        args.file = [path.rstrip("/\\") for path in args.file if path]

    gpu_name_display = None
    gpu_vram_gb = None
    gpu_count = 1
    
    if args.gpu or args.vllm:
        target = args.gpu if args.gpu else "auto"
        from modelinfo.hardware import resolve_gpu
        gpu_name_display, gpu_vram_gb, gpu_count = resolve_gpu(target)

    if len(args.file) > 1:
        if args.vllm:
            error_console.print("[red]Error: Side-by-side comparison does not currently support the --vllm capacity simulation. Compare models sequentially or remove --vllm.[/red]")
            return 1
            
        models = []
        for model_path in args.file:
            info = analyze_model(
                model_path, 
                args.context, 
                gpu_count=gpu_count,
                batch_size=args.batch_size,
                fetch_tensors=args.tensors,
                timeout=args.timeout,
                topology=args.topology,
                strategy=args.strategy,
                is_vllm=args.vllm,
                gpu_vram_gb=gpu_vram_gb if gpu_vram_gb else 0.0,
                gpu_util=args.gpu_util
            )
            models.append((model_path.split("/")[-1], info))
            
        max_vram_gb = gpu_vram_gb if gpu_vram_gb else args.max_vram
        if args.json:
            reports = [
                {
                    "name": name,
                    "info": _json_report(info, gpu_name_display, max_vram_gb, args.tensors),
                }
                for name, info in models
            ]
            print(json.dumps(reports, indent=2))
        else:
            print_compare_info(models, max_vram_gb, gpu_name=gpu_name_display)
        return 0
        
    file_path = args.file[0]
    
    info = analyze_model(
        file_path, 
        args.context, 
        gpu_count=gpu_count,
        batch_size=args.batch_size,
        fetch_tensors=args.tensors,
        timeout=args.timeout,
        topology=args.topology,
        strategy=args.strategy,
        is_vllm=args.vllm,
        gpu_vram_gb=gpu_vram_gb if gpu_vram_gb else 0.0,
        gpu_util=args.gpu_util
    )

    max_vram_gb = gpu_vram_gb if gpu_vram_gb else args.max_vram
    if args.json:
        report = _json_report(info, gpu_name_display, max_vram_gb, args.tensors)
        print(json.dumps(report, indent=2))
    else:
        print_model_info(**info, max_vram_gb=max_vram_gb, gpu_name=gpu_name_display)
    return 0


if __name__ == "__main__":
    sys.exit(main())
