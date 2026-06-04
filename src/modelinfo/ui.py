import math
import re
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from modelinfo.calculator import format_bytes, format_params

console = Console()

def get_vram_color(bytes_size: float) -> str:
    gb = bytes_size / (1024**3)
    if gb < 8.0:
        return "green"
    elif gb < 16.0:
        return "yellow"
    else:
        return "red"

def group_tensors_by_size(tensors: Dict[str, Any]):
    groups = {}
    for name, metadata in tensors.items():
        if name == "__metadata__":
            continue
            
        shape = tuple(metadata.get("shape", []))
        dtype = metadata.get("dtype", "Unknown")
        
        base_name = re.sub(r"\.\d+\.", ".[N].", name)
        key = (base_name, shape, dtype)
        
        if key not in groups:
            groups[key] = {"count": 0, "params": math.prod(shape) if shape else 0}
        groups[key]["count"] += 1
        
    return sorted(groups.items(), key=lambda x: x[1]["params"], reverse=True)

def print_model_info(
    format_name: str,
    arch_name: str,
    tensor_count: int,
    footprint: Dict[str, Any],
    disk_size: float,
    context_length: int,
    tensors: Dict[str, Any],
    max_context: int | None = None
) -> None:
    summary = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    summary.add_column("Property", style="bold")
    summary.add_column("Value")
    
    metadata = tensors.get("__metadata__", {})
    missing_shards = metadata.get("missing_shards", 0)
    total_shards = metadata.get("total_shards", 0)
    
    if missing_shards > 0:
        param_text = "[yellow]UNKNOWN (Missing Shards)[/yellow]"
        disk_text = "[yellow]UNKNOWN (Missing Shards)[/yellow]"
        vram_display = "[yellow]UNKNOWN (Missing Shards)[/yellow]"
    else:
        param_text = format_params(footprint["total_params"])
        disk_text = format_bytes(disk_size)
        vram_bytes = footprint["total_memory_bytes"]
        vram_color = "green" if vram_bytes < 8 * 1024**3 else "yellow" if vram_bytes < 16 * 1024**3 else "red"
        
        vram_text = f"~{format_bytes(vram_bytes)}"
        if context_length > 0:
            if footprint.get("kv_is_estimate"):
                vram_text += f" ({footprint['primary_dtype']}, Estimated KV Cache - Missing Config)"
            else:
                vram_text += f" ({footprint['primary_dtype']}, KV cache for {context_length} tokens)"
        else:
            vram_text += f" ({footprint['primary_dtype']}, no KV cache)"
        vram_display = f"[{vram_color}]{vram_text}[/{vram_color}]"

    summary.add_row("Format:", format_name)
    summary.add_row("Architecture:", arch_name)
    summary.add_row("Tensors:", f"{tensor_count:,}")
    summary.add_row("Parameters:", param_text)
    summary.add_row("Dtype:", footprint["primary_dtype"])
    summary.add_row("Disk size:", disk_text)
    summary.add_row("VRAM (est):", vram_display)
    
    console.print(summary)

    if missing_shards > 0:
        console.print(f"[bold yellow]WARNING: Partial Model. Missing {missing_shards} of {total_shards} shards on disk. Totals are incomplete.[/bold yellow]")
        
    if context_length > 0 and max_context is not None and context_length > max_context:
        console.print(f"[bold yellow]WARNING: Requested context ({context_length:,}) exceeds model's native limit ({max_context:,}).[/bold yellow]")

    console.print()
    
    console.print("Top Tensors by Size:", style="bold")
    
    grouped_tensors = group_tensors_by_size(tensors)
    
    tensor_table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    tensor_table.add_column("Name", no_wrap=True, overflow="fold")
    tensor_table.add_column("Shape", justify="right")
    tensor_table.add_column("Dtype", justify="left")
    tensor_table.add_column("Params", justify="right")
    
    for i, (key, data) in enumerate(grouped_tensors):
        if i >= 5:
            break
            
        base_name, shape, dtype = key
        count = data["count"]
        params = data["params"]
        
        display_name = f"  {count}x {base_name}" if count > 1 else f"  {base_name}"
        shape_str = f"[{' x '.join(map(str, shape))}]" if shape else "[]"
        param_str = format_params(params).split(' ')[-1].replace("(", "").replace(")", "") + " params"
        if not param_str[0].isdigit():
            param_str = str(params) + " params"
            
        tensor_table.add_row(display_name, shape_str, dtype.lower(), param_str)
        
    console.print(tensor_table)
