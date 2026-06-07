import struct
from typing import Any, Dict

GGML_TYPE_MAP = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_O", 5: "Q4_0_O", 
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 
    29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
}

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
            
            # Strict GGUF tensor type mapping
            dtype = GGML_TYPE_MAP.get(t_type, "Unknown")
                
            tensors[name] = {"shape": shape, "dtype": dtype}
            
    return tensors
