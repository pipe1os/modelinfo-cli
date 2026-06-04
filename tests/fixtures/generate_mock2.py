import struct
import pickle
import zipfile

class DummyStorage:
    def __init__(self, name):
        self.name = name

def make_dummy_storage(name):
    return DummyStorage(name)

with zipfile.ZipFile("mock_model.pt", "w") as zf:
    data = {
        "model.layers.0.self_attn.k_proj.weight": {
            "shape": [1024, 4096],
            "dtype": "F16"
        }
    }
    pkl_bytes = pickle.dumps(data)
    zf.writestr("archive/data.pkl", pkl_bytes)

with open("mock_model.gguf", "wb") as f:
    f.write(b"GGUF")
    f.write(struct.pack("<I", 3)) # version
    f.write(struct.pack("<Q", 1)) # tensor count
    f.write(struct.pack("<Q", 5)) # kv count
    
    # Write general.architecture metadata
    key = b"general.architecture"
    f.write(struct.pack("<Q", len(key)))
    f.write(key)
    f.write(struct.pack("<I", 8)) # val_type 8 = string
    val = b"qwen2"
    f.write(struct.pack("<Q", len(val)))
    f.write(val)
    
    # Write qwen2.block_count
    key = b"qwen2.block_count"
    f.write(struct.pack("<Q", len(key)))
    f.write(key)
    f.write(struct.pack("<I", 4)) # val_type 4 = uint32
    f.write(struct.pack("<I", 32))
    
    # Write qwen2.attention.head_count_kv
    key = b"qwen2.attention.head_count_kv"
    f.write(struct.pack("<Q", len(key)))
    f.write(key)
    f.write(struct.pack("<I", 4)) # val_type 4 = uint32
    f.write(struct.pack("<I", 8))
    
    # Write qwen2.attention.key_length
    key = b"qwen2.attention.key_length"
    f.write(struct.pack("<Q", len(key)))
    f.write(key)
    f.write(struct.pack("<I", 4)) # val_type 4 = uint32
    f.write(struct.pack("<I", 128))
    # Write qwen2.context_length
    key = b"qwen2.context_length"
    f.write(struct.pack("<Q", len(key)))
    f.write(key)
    f.write(struct.pack("<I", 4)) # val_type 4 = uint32
    f.write(struct.pack("<I", 32768))
    
    name = b"model.layers.0.self_attn.k_proj.weight"
    f.write(struct.pack("<Q", len(name)))
    f.write(name)
    
    f.write(struct.pack("<I", 2)) # 2 dims
    f.write(struct.pack("<Q", 4096))
    f.write(struct.pack("<Q", 1024))
    
    f.write(struct.pack("<I", 1)) # F16
    f.write(struct.pack("<Q", 0)) # offset
