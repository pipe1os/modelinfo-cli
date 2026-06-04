import struct
import json

header = {
    "__metadata__": {
        "format": "pt"
    },
    "model.embed_tokens.weight": {
        "dtype": "BF16",
        "shape": [32000, 4096],
        "data_offsets": [0, 262144000]
    }
}

offset = 262144000
for i in range(32):
    q_shape = [4096, 4096]
    q_size = 4096 * 4096 * 2
    header[f"model.layers.{i}.self_attn.q_proj.weight"] = {
        "dtype": "BF16",
        "shape": q_shape,
        "data_offsets": [offset, offset + q_size]
    }
    offset += q_size
    
    k_shape = [1024, 4096]
    k_size = 1024 * 4096 * 2
    header[f"model.layers.{i}.self_attn.k_proj.weight"] = {
        "dtype": "BF16",
        "shape": k_shape,
        "data_offsets": [offset, offset + k_size]
    }
    offset += k_size

json_bytes = json.dumps(header).encode("utf-8")
header_length = len(json_bytes)

with open("mock_mistral-7b.safetensors", "wb") as f:
    f.write(struct.pack("<Q", header_length))
    f.write(json_bytes)
    f.write(b"0" * 1024)
