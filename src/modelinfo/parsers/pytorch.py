import pickle
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
        },
    }

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


def parse_pytorch_header(path: str) -> Dict[str, Any]:
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
        for k, v in data.items():
            if isinstance(v, dict) and "shape" in v:
                tensors[k] = v
            else:
                tensors[k] = {"shape": [], "dtype": "F32"}
                
    return tensors
