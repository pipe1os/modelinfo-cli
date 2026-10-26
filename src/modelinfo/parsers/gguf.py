import struct
from typing import Any

GGML_TYPE_MAP = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_O", 5: "Q4_0_O", 
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 
    12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 
    29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

# Lengths come from the file, so a corrupt one must hit EOF instead of allocating it.
_MAX_READ_CHUNK = 1024 * 1024

# Fixed-size GGUF value types; 8 (string) and 9 (array) are handled separately.
_SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
    10: "Q", 11: "q", 12: "d",
}


def _read_exact(f: Any, size: int) -> bytes:
    chunk = f.read(min(size, _MAX_READ_CHUNK))
    if len(chunk) == size:
        return chunk
    data = bytearray(chunk)
    while len(data) < size:
        chunk = f.read(min(size - len(data), _MAX_READ_CHUNK))
        if not chunk:
            raise EOFError("Unexpected end of GGUF header")
        data.extend(chunk)
    return bytes(data)


def _read_gguf_value(f: Any, val_type: int) -> Any:
    fmt = _SCALAR_FORMATS.get(val_type)
    if fmt is not None:
        return struct.unpack("<" + fmt, _read_exact(f, struct.calcsize(fmt)))[0]
    if val_type == 8:
        slen = struct.unpack("<Q", _read_exact(f, 8))[0]
        return _read_exact(f, slen).decode("utf-8")
    if val_type == 9:
        arr_type = struct.unpack("<I", _read_exact(f, 4))[0]
        arr_len = struct.unpack("<Q", _read_exact(f, 8))[0]
        fmt = _SCALAR_FORMATS.get(arr_type)
        if fmt is not None:
            # Tokenizer scores and token types are 100k+ numbers; unpack them in one call.
            raw = _read_exact(f, arr_len * struct.calcsize(fmt))
            return list(struct.unpack(f"<{arr_len}{fmt}", raw))
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    raise ValueError(f"Unknown GGUF value type: {val_type}")


def parse_gguf_header(path_or_file: str | Any) -> dict[str, Any]:
    """Parses a GGUF file header and extracts tensor information."""
    if isinstance(path_or_file, str):
        with open(path_or_file, "rb") as f:
            return _parse_gguf_header_from_stream(f)
    else:
        return _parse_gguf_header_from_stream(path_or_file)


def _parse_gguf_header_from_stream(f: Any) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    magic = _read_exact(f, 4)
    if magic != b"GGUF":
        raise ValueError("Invalid GGUF file: Magic bytes missing.")
        
    version = struct.unpack("<I", _read_exact(f, 4))[0]
    if version < 2:
        raise ValueError(f"Unsupported GGUF version: {version}")
        
    tensor_count = struct.unpack("<Q", _read_exact(f, 8))[0]
    kv_count = struct.unpack("<Q", _read_exact(f, 8))[0]
    
    metadata = {}
    for _ in range(kv_count):
        key_len = struct.unpack("<Q", _read_exact(f, 8))[0]
        key_name = _read_exact(f, key_len).decode("utf-8")
        val_type = struct.unpack("<I", _read_exact(f, 4))[0]
        metadata[key_name] = _read_gguf_value(f, val_type)
        
    tensors["__metadata__"] = metadata
        
    for _ in range(tensor_count):
        name_len = struct.unpack("<Q", _read_exact(f, 8))[0]
        name = _read_exact(f, name_len).decode("utf-8")
        
        n_dims = struct.unpack("<I", _read_exact(f, 4))[0]
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack("<Q", _read_exact(f, 8))[0])
        
        t_type = struct.unpack("<I", _read_exact(f, 4))[0]
        _read_exact(f, 8)  # skip offset bytes
        
        # Strict GGUF tensor type mapping
        dtype = GGML_TYPE_MAP.get(t_type, "Unknown")
            
        tensors[name] = {"shape": shape, "dtype": dtype}
        
    return tensors

