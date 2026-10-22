import json
import os
import pickle
import re
import zipfile
from typing import Any, Dict


class DummyStorage:
    def __init__(self, module: str, name: str):
        self.module = module
        self.name = name


def dummy_rebuild_tensor_v2(
    storage: Any,
    storage_offset: Any,
    size: Any,
    stride: Any,
    requires_grad: Any,
    backward_hooks: Any,
    metadata: Any = None,
) -> Dict[str, Any]:
    dtype = "F32"
    if isinstance(storage, DummyStorage):
        if storage.name == "HalfStorage":
            dtype = "F16"
        elif storage.name == "BFloat16Storage":
            dtype = "BF16"
        elif storage.name == "IntStorage":
            dtype = "I32"
        elif storage.name == "LongStorage":
            dtype = "I64"
        elif storage.name == "DoubleStorage":
            dtype = "F64"
        elif storage.name == "ShortStorage":
            dtype = "I16"
        elif storage.name == "CharStorage":
            dtype = "I8"
        elif storage.name == "ByteStorage":
            dtype = "U8"
        elif storage.name == "BoolStorage":
            dtype = "BOOL"
    return {"shape": list(size), "dtype": dtype}


class RestrictedUnpickler(pickle.Unpickler):
    """
    A secure unpickler that only allows basic Python types and
    dummy PyTorch primitives to be instantiated.
    Blocks arbitrary code execution.
    """
    ALLOWED_MODULES = {
        "collections": {"OrderedDict"},
        "torch._utils": {"_rebuild_tensor_v2"},
        "torch": {
            "FloatStorage",
            "HalfStorage",
            "BFloat16Storage",
            "IntStorage",
            "LongStorage",
            "DoubleStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
        },
    }

    def persistent_load(self, saved_id: Any) -> DummyStorage:
        # ZIP checkpoints refer to tensor storage via a five-element ID.
        # Only reconstruct metadata using our existing dummy allowlist; never
        # load storage bytes, import torch, or invoke a pickle-provided callable.
        if not isinstance(saved_id, tuple) or len(saved_id) != 5:
            raise pickle.UnpicklingError("Invalid storage persistent ID")
        kind, storage_type, _, _, _ = saved_id
        if (kind != "storage" or not isinstance(storage_type, type)
                or not issubclass(storage_type, DummyStorage)
                or storage_type is DummyStorage):
            raise pickle.UnpicklingError("Unsupported storage persistent ID")
        return storage_type()

    def find_class(self, module: str, name: str) -> Any:
        if module in self.ALLOWED_MODULES and name in self.ALLOWED_MODULES[module]:
            if name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            if name == "_rebuild_tensor_v2":
                return dummy_rebuild_tensor_v2
            return type(
                f"Dummy_{name}",
                (DummyStorage,),
                {
                    "__init__": lambda self, *args, **kwargs: DummyStorage.__init__(
                        self, module, name
                    )
                },
            )
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden for security reasons.")


def _read_single_pytorch_header(path: str) -> Dict[str, Any]:
    tensors = {}

    if not zipfile.is_zipfile(path):
        raise ValueError("PyTorch file is not a valid zip archive (legacy format not supported).")
        
    with zipfile.ZipFile(path, "r") as zf:
        pkl_names = [name for name in zf.namelist() if name.endswith("data.pkl")]
        if not pkl_names:
            raise ValueError("Could not find data.pkl in PyTorch zip archive.")
            
        with zf.open(pkl_names[0], "r") as pf:
            unpickler = RestrictedUnpickler(pf)
            data = unpickler.load()
            
    if isinstance(data, dict):
        for key in ("state_dict", "model_state_dict"):
            candidate = data.get(key)
            if isinstance(candidate, dict) and "shape" not in candidate:
                data = candidate
                break
        for k, v in data.items():
            if isinstance(v, dict) and "shape" in v:
                tensors[k] = v
            else:
                tensors[k] = {"shape": [], "dtype": "F32"}

    return tensors


def parse_pytorch_index(index_path: str) -> Dict[str, Any]:
    """Merge the shards listed in a ``*.bin.index.json`` / ``*.pt.index.json``.

    Mirrors the SafeTensors index path in ``parse_safetensors_header``: only
    tensors the ``weight_map`` assigns to a shard are taken from it, shards
    absent on disk are counted instead of failing, and the result carries
    the same ``__metadata__`` keys that path sets.
    """
    dir_path = os.path.dirname(index_path)

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    unique_shards = set(weight_map.values())

    tensors: Dict[str, Any] = {}
    missing_shards = 0
    total_shards = len(unique_shards)
    total_size = 0

    for shard in unique_shards:
        shard_path = os.path.join(dir_path, shard)
        if not os.path.exists(shard_path):
            missing_shards += 1
            continue
        total_size += os.path.getsize(shard_path)
        shard_header = _read_single_pytorch_header(shard_path)
        for name, assigned_shard in weight_map.items():
            if assigned_shard != shard:
                continue
            if name not in shard_header:
                raise ValueError(
                    f"Indexed tensor {name!r} is missing from shard {shard!r}"
                )
            tensors[name] = shard_header[name]

    tensors["__metadata__"] = {
        "missing_shards": missing_shards,
        "total_shards": total_shards,
        "is_sharded": True,
        "disk_size": total_size,
    }

    return tensors


def parse_pytorch_header(path: str) -> Dict[str, Any]:
    if path.endswith(".index.json"):
        return parse_pytorch_index(path)

    base_name = os.path.basename(path)
    if "-of-" in base_name and path.lower().endswith((".bin", ".pt")):
        match = re.match(r"^(.*?)-\d+-of-\d+(\.bin|\.pt)$", base_name, re.IGNORECASE)
        if match:
            prefix, extension = match.group(1), match.group(2)
            potential_index = os.path.join(
                os.path.dirname(path), f"{prefix}{extension}.index.json"
            )
            if os.path.exists(potential_index):
                return parse_pytorch_index(potential_index)

    return _read_single_pytorch_header(path)
