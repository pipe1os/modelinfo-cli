import json
import struct
from typing import Any

import os

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
            
        return json.loads(json_bytes)

def parse_safetensors_header(path: str) -> dict[str, Any]:
    dir_path = os.path.dirname(path)
    base_name = os.path.basename(path)
    
    index_path = path
    is_index = False
    
    if path.endswith(".index.json"):
        is_index = True
    elif "-of-" in base_name and path.endswith(".safetensors"):
        import re
        match = re.match(r"^(.*?)-\d{5}-of-\d{5}\.safetensors$", base_name)
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
    
    for shard in unique_shards:
        shard_path = os.path.join(dir_path, shard)
        try:
            shard_header = _read_single_header(shard_path)
            for k, v in shard_header.items():
                if k != "__metadata__":
                    tensors[k] = v
        except FileNotFoundError:
            missing_shards += 1
            
    tensors["__metadata__"] = {
        "missing_shards": missing_shards,
        "total_shards": total_shards,
        "is_sharded": True
    }
    
    return tensors
