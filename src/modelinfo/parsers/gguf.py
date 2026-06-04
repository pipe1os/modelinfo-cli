import struct
from typing import Any, Dict


def _read_gguf_value(f: Any, val_type: int) -> Any:
    if val_type == 0:
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == 1:
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == 2:
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == 3:
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == 4:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == 5:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == 6:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == 7:
        return struct.unpack("<?", f.read(1))[0]
    elif val_type == 8:
        slen = struct.unpack("<Q", f.read(8))[0]
        return f.read(slen).decode("utf-8")
    elif val_type == 9:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    elif val_type == 10:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == 11:
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == 12:
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {val_type}")


def parse_gguf_header(path: str) -> Dict[str, Any]:
    """Parses a GGUF file header and extracts tensor information."""
    tensors: Dict[str, Any] = {}
    
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("Invalid GGUF file: Magic bytes missing.")
            
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"Unsupported GGUF version: {version}")
            
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        
        metadata = {}
        for _ in range(kv_count):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key_name = f.read(key_len).decode("utf-8")
            val_type = struct.unpack("<I", f.read(4))[0]
            metadata[key_name] = _read_gguf_value(f, val_type)
            
        tensors["__metadata__"] = metadata
            
        for _ in range(tensor_count):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack("<Q", f.read(8))[0])
            
            t_type = struct.unpack("<I", f.read(4))[0]
            f.read(8)  # skip offset bytes
            
            # Simplified GGUF tensor type mapping
            dtype = "F32"
            if t_type == 1:
                dtype = "F16"
            elif t_type > 1:
                dtype = "Q4" # Generic placeholder for quantized types
                
            tensors[name] = {"shape": shape, "dtype": dtype}
            
    return tensors
