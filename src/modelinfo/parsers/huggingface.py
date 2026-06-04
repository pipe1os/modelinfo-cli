import concurrent.futures
import json
import os
import struct
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

def _get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
        
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            pass
    return None

def _make_request(url: str, headers: Dict[str, str] = None) -> bytes:
    if headers is None:
        headers = {}
        
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(f"Gated Model or Invalid Token: Please set HF_TOKEN environment variable to access {url}")
        if e.code == 404:
            raise FileNotFoundError(f"File not found on Hugging Face Hub: {url}")
        raise

def _fetch_safetensors_header(repo_id: str, filename: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    # 1. Fetch the first 500KB in a single roundtrip
    headers = {"Range": "bytes=0-500000"}
    try:
        chunk = _make_request(url, headers=headers)
    except urllib.error.HTTPError as e:
        if e.code == 416: # Range Not Satisfiable (file is smaller than 500KB)
            chunk = _make_request(url)
        else:
            raise
            
    if len(chunk) < 8:
        raise ValueError(f"File {filename} is too small to contain a SafeTensors header.")
        
    header_size = struct.unpack("<Q", chunk[:8])[0]
    
    # 2. Slice locally if it fits
    if 8 + header_size <= len(chunk):
        json_bytes = chunk[8:8+header_size]
    else:
        # 3. Double-roundtrip only if the header is massive (>500KB)
        headers = {"Range": f"bytes=8-{8+header_size-1}"}
        json_bytes = _make_request(url, headers=headers)
        
    return json.loads(json_bytes)

def fetch_huggingface_repo(repo_id: str) -> Tuple[Dict[str, Any], Dict[str, Any] | None, str, float]:
    """
    Fetches the metadata directly from the Hugging Face Hub over the network.
    Returns: (tensors, config, format_name, disk_size)
    """
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        api_data = json.loads(_make_request(api_url).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(f"Gated Model: Please set HF_TOKEN environment variable to access {repo_id}")
        raise
        
    siblings = api_data.get("siblings", [])
    filenames = {s["rfilename"] for s in siblings}
    
    config = None
    if "config.json" in filenames:
        config_url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
        config = json.loads(_make_request(config_url).decode("utf-8"))
        
    tensors = {}
    total_size = 0.0
    
    if "model.safetensors.index.json" in filenames:
        # Sharded SafeTensors
        index_url = f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors.index.json"
        index_data = json.loads(_make_request(index_url).decode("utf-8"))
        
        weight_map = index_data.get("weight_map", {})
        unique_shards = list(set(weight_map.values()))
        
        total_size = index_data.get("metadata", {}).get("total_size", 0.0)
        
        def fetch_shard(shard: str):
            return shard, _fetch_safetensors_header(repo_id, shard)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(unique_shards))) as executor:
            future_to_shard = {executor.submit(fetch_shard, shard): shard for shard in unique_shards}
            for future in concurrent.futures.as_completed(future_to_shard):
                shard, shard_header = future.result()
                for k, v in shard_header.items():
                    if k != "__metadata__":
                        tensors[k] = v
                        
        tensors["__metadata__"] = {
            "missing_shards": 0,
            "total_shards": len(unique_shards),
            "is_sharded": True
        }
        format_name = "SafeTensors"
        
    elif "model.safetensors" in filenames:
        # Single SafeTensors
        header = _fetch_safetensors_header(repo_id, "model.safetensors")
        tensors = header
        format_name = "SafeTensors"
        
        # We don't have total_size from index, so we could get it from Content-Length or just leave it 0
        req = urllib.request.Request(f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors", method="HEAD")
        token = _get_hf_token()
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))
        except Exception:
            pass
            
    else:
        raise ValueError(f"Repository {repo_id} does not contain SafeTensors weights.")
        
    return tensors, config, format_name, float(total_size)
