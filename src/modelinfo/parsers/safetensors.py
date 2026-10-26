import json
import os
import struct
from typing import Any


def _read_single_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        header_length_bytes = f.read(8)
        if len(header_length_bytes) != 8:
            raise EOFError("Invalid SafeTensors file: Unexpected end of file while reading header length.")
            
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        
        if header_length > 100 * 1024 * 1024:
            raise ValueError(f"Header length ({header_length} bytes) exceeds maximum allowed size.")
            
        json_bytes = f.read(header_length)
        if len(json_bytes) != header_length:
            raise EOFError("Invalid SafeTensors file: Unexpected end of file while reading JSON header.")
            
        header = json.loads(json_bytes)
        if not isinstance(header, dict):
            raise ValueError(f"SafeTensors header in {path} must be a JSON object.")  # noqa: TRY004 - ValueError is the documented error for bad headers
        return header

def parse_safetensors_header(path: str) -> dict[str, Any]:
    dir_path = os.path.dirname(path)
    base_name = os.path.basename(path)
    
    index_path = path
    is_index = False
    
    if path.endswith(".index.json"):
        is_index = True
    elif "-of-" in base_name and path.endswith(".safetensors"):
        import re
        match = re.match(r"^(.*?)-\d+-of-\d+\.safetensors$", base_name)
        if match:
            prefix = match.group(1)
        else:
            # Fallback to splitting in case of non-standard shard formatting
            prefix = base_name.split("-")[0]
        potential_index = os.path.join(dir_path, f"{prefix}.safetensors.index.json")
        if os.path.exists(potential_index):
            index_path = potential_index
            is_index = True
    
    if not is_index:
        return _read_single_header(path)
        
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    unique_shards = set(weight_map.values())
    
    tensors = {}
    missing_shards = 0
    total_shards = len(unique_shards)
    total_size = 0
    
    for shard in unique_shards:
        shard_path = os.path.join(dir_path, shard)
        if os.path.exists(shard_path):
            total_size += os.path.getsize(shard_path)
        try:
            shard_header = _read_single_header(shard_path)
            for name, assigned_shard in weight_map.items():
                if assigned_shard != shard:
                    continue
                if name not in shard_header:
                    raise ValueError(f"Indexed tensor {name!r} is missing from shard {shard!r}")
                tensors[name] = shard_header[name]
        except FileNotFoundError:
            missing_shards += 1
            
    tensors["__metadata__"] = {
        "missing_shards": missing_shards,
        "total_shards": total_shards,
        "is_sharded": True,
        "disk_size": total_size
    }
    
    return tensors
