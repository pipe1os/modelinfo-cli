import math
import re
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from modelinfo.calculator import format_bytes, format_params

console = Console()

def get_vram_color(bytes_size: float, max_vram_gb: float = 8.0) -> str:
    gb = bytes_size / (1024**3)
    if gb <= max_vram_gb:
        return "green"
    elif gb <= max_vram_gb * 2:
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
    is_default_context: bool,
    tensors: Dict[str, Any],
    max_context: int | None = None,
    max_vram_gb: float = 8.0,
    gpu_name: str | None = None,
    is_lazy: bool = False,
    gpu_count: int = 1,
    topology: str = "pcie4",
    strategy: str = "tp",
    is_vllm: bool = False,
    gpu_vram_gb: float = 0.0,
    gpu_util: float = 0.9
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
    elif footprint["total_memory_bytes"] == 0:
        param_text = format_params(footprint["total_params"])
        disk_text = format_bytes(disk_size)
        vram_display = "[red]Unknown (Missing Tensor Shapes)[/red]"
    else:
        param_text = format_params(footprint["total_params"])
        disk_text = format_bytes(disk_size)
        vram_bytes = footprint["total_memory_bytes"]
        vram_color = get_vram_color(vram_bytes, max_vram_gb)
        
        vram_text = f"~{format_bytes(vram_bytes)} Total Minimum Required"
        vram_display = f"[{vram_color}]{vram_text}[/{vram_color}]\n"
        
        weights_bytes = footprint["base_memory_bytes"]
        vram_display += f"  ├─ Weights:    {format_bytes(weights_bytes)}\n"
        
        kv_cache_bytes = footprint["kv_cache_bytes"]
        
        if footprint.get("kv_is_estimate"):
            kv_note = " (Estimated KV Cache - Missing Config)"
        elif is_default_context:
            if max_context and max_context > context_length:
                kv_note = f" (Default {context_length} tokens. Native limit: {max_context:,})"
            else:
                kv_note = f" (Default {context_length} tokens)"
        else:
            kv_note = f" ({context_length} tokens)"
            
        vram_display += f"  ├─ KV Cache:   {format_bytes(kv_cache_bytes)}{kv_note}\n"
        
        overhead_bytes = footprint.get("overhead_bytes", 600 * 1024 * 1024)
        if gpu_count > 1:
            penalty_str = f"TP/{topology}" if strategy == "tp" else "PP"
            vram_display += f"  └─ Overhead:   {format_bytes(overhead_bytes)} (CUDA Contexts + {penalty_str} Penalty)"
        else:
            vram_display += f"  └─ Overhead:   {format_bytes(overhead_bytes)} (CUDA Context + Activations)"

    summary.add_row("Format:", format_name)
    summary.add_row("Architecture:", arch_name)
    summary.add_row("Tensors:", f"{tensor_count:,}")
    summary.add_row("Parameters:", param_text)
    summary.add_row("Dtype:", footprint["primary_dtype"])
    summary.add_row("Disk size:", disk_text)
    
    if is_vllm:
        vllm = footprint.get("vllm_metrics", {})
        usable_vram = vllm.get("usable_vram", 0.0)
        static_weights = vllm.get("static_weights", 0.0)
        distributed_penalty = vllm.get("distributed_penalty", 0.0)
        paged_kv_pool = vllm.get("paged_kv_pool", 0.0)
        max_capacity = vllm.get("max_serving_capacity", 0)
        
        summary.add_row("VRAM Ceiling:", f"{max_vram_gb:.1f} GB ({gpu_name if gpu_name else 'Target'})")
        
        alloc_display = f"  ├─ Usable VRAM:      {format_bytes(usable_vram)} ({int(gpu_util*100)}% gpu_memory_utilization)\n"
        alloc_display += f"  ├─ Static Weights:  -{format_bytes(static_weights)} ({footprint.get('primary_dtype', 'BF16')})\n"
        if gpu_count > 1:
            penalty_str = f"TP/{topology}" if strategy == "tp" else "PP"
            alloc_display += f"  ├─ {penalty_str} Penalty: -{format_bytes(distributed_penalty)}\n"
        alloc_display += f"  └─ Paged KV Pool:   = {format_bytes(paged_kv_pool)} Available for Context"
        
        summary.add_row("vLLM Allocation:", alloc_display)
        summary.add_row("Max Capacity:", f"~{max_capacity:,} Tokens (Across all concurrent batches)")
        
        if paged_kv_pool <= 0:
            summary.add_row("Hardware Fit:", "[red]✗ No (OOM before serving any tokens)[/red]")
        else:
            summary.add_row("Hardware Fit:", "[green]✓ Yes[/green]")
    else:
        summary.add_row("VRAM (est):", vram_display)
        if gpu_name:
            utilization = vram_bytes / (max_vram_gb * 1024**3) if max_vram_gb > 0 else 2.0
            if utilization <= 0.90:
                fit_text = f"[green]✓ Fits comfortably in {gpu_name} ({max_vram_gb:.1f} GB)[/green]"
            elif utilization <= 0.99:
                fit_text = f"[yellow]⚠ Warning: Extreme hardware limit on {gpu_name}. High risk of fragmentation OOM.[/yellow]"
            else:
                fit_text = f"[red]✗ No (Requires {format_bytes(vram_bytes)}, Hardware has {max_vram_gb:.1f} GB)[/red]"
            summary.add_row("Hardware Fit:", fit_text)
    
    console.print(summary)

    if missing_shards > 0:
        console.print(f"[bold yellow]WARNING: Partial Model. Missing {missing_shards} of {total_shards} shards on disk. Totals are incomplete.[/bold yellow]")
        
    if context_length > 0 and max_context is not None and context_length > max_context:
        console.print(f"[bold yellow]WARNING: Requested context ({context_length:,}) exceeds model's native limit ({max_context:,}).[/bold yellow]")

    console.print()
    
    if is_lazy:
        console.print("[yellow]Top Tensors omitted for speed. Run with --tensors to fetch remote shards.[/yellow]")
        return
        
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

def print_compare_info(models: list, max_vram_gb: float = 8.0, gpu_name: str | None = None) -> None:
    table = Table(box=None, show_header=True, header_style="bold", pad_edge=False, padding=(0, 2))
    table.add_column("Model")
    table.add_column("Params")
    table.add_column("Dtype")
    table.add_column("Context")
    table.add_column("VRAM")
    if gpu_name:
        table.add_column("Fits")
    
    for name, info in models:
        footprint = info["footprint"]
        
        param_full = format_params(footprint["total_params"])
        if "(" in param_full:
            param_text = param_full.split("(")[-1].replace(")", "")
        else:
            param_text = param_full
            
        dtype_text = footprint.get("primary_dtype", "Unknown")
            
        context_length = info.get("context_length", 0)
        if context_length >= 1024 and context_length % 1024 == 0:
            context_text = f"{context_length // 1024}K"
        else:
            context_text = f"{context_length:,}"
            
        vram_bytes = footprint["total_memory_bytes"]
        vram_color = get_vram_color(vram_bytes, max_vram_gb)
        vram_text = f"[{vram_color}]{format_bytes(vram_bytes)}[/{vram_color}]"
        
        row = [name, param_text, dtype_text, context_text, vram_text]
        
        if gpu_name:
            utilization = vram_bytes / (max_vram_gb * 1024**3) if max_vram_gb > 0 else 2.0
            if utilization <= 0.90:
                row.append("[green]✓[/green]")
            elif utilization <= 0.99:
                row.append("[yellow]⚠[/yellow]")
            else:
                row.append("[red]✗[/red]")
                
        table.add_row(*row)
        
    console.print(table)

