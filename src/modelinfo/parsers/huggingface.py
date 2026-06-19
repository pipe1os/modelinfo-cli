import concurrent.futures
import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple


def _get_hf_endpoint() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").strip()
    if not endpoint:
        raise ValueError("HF_ENDPOINT is set but empty; expected a valid HTTP(S) URL")
    endpoint = endpoint.rstrip("/")
    if not endpoint.startswith("https://"):
        raise ValueError(
            f"HF_ENDPOINT must use https:// scheme, got: {endpoint}"
        )
    parsed = urllib.parse.urlparse(endpoint)
    if not parsed.netloc:
        raise ValueError(
            f"HF_ENDPOINT must include a valid hostname, got: {endpoint}"
        )
    return endpoint


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

    legacy_path = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            pass

    return None


def _make_request(
    url: str,
    headers: Dict[str, str] = None,
    limit: int | None = None,
    timeout: float = 10.0,
) -> bytes:
    if headers is None:
        headers = {}

    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if limit is not None:
                return response.read(limit)
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(
                f"Gated/Private Model or Invalid Token (401 Unauthorized). Set the HF_TOKEN environment variable to access {url}"
            )
        if e.code == 404:
            raise FileNotFoundError(
                f"Could not find repository or file on Hugging Face (404 Not Found): {url}"
            )
        raise


def _fetch_safetensors_header(
    repo_id: str, filename: str, timeout: float = 10.0
) -> Dict[str, Any]:
    url = f"{_get_hf_endpoint()}/{repo_id}/resolve/main/{filename}"

    # 1. Fetch the first 500KB in a single roundtrip
    headers = {"Range": "bytes=0-500000"}
    try:
        chunk = _make_request(url, headers=headers, limit=500000, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable (file is smaller than 500KB)
            chunk = _make_request(url, limit=500000, timeout=timeout)
        else:
            raise

    if len(chunk) < 8:
        raise ValueError(
            f"File {filename} is too small to contain a SafeTensors header."
        )

    header_size = struct.unpack("<Q", chunk[:8])[0]

    # 2. Slice locally if it fits
    if 8 + header_size <= len(chunk):
        json_bytes = chunk[8 : 8 + header_size]
    else:
        # 3. Double-roundtrip only if the header is massive (>500KB)
        headers = {"Range": f"bytes=8-{8 + header_size - 1}"}
        json_bytes = _make_request(
            url, headers=headers, limit=header_size, timeout=timeout
        )

    return json.loads(json_bytes)


def fetch_huggingface_repo(
    repo_id: str, fetch_tensors: bool = False, timeout: float = 10.0
) -> Tuple[Dict[str, Any], Dict[str, Any] | None, str, float]:
    """
    Fetches the metadata directly from the Hugging Face Hub over the network.
    Returns: (tensors, config, format_name, disk_size)
    """
    api_url = f"{_get_hf_endpoint()}/api/models/{repo_id}"
    try:
        api_data = json.loads(_make_request(api_url, timeout=timeout).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(
                f"Gated/Private Model (401 Unauthorized). Set the HF_TOKEN environment variable to access {repo_id}"
            )
        if e.code == 404:
            raise FileNotFoundError(
                f"Could not find repository on Hugging Face (404 Not Found): {repo_id}"
            )
        raise

    siblings = api_data.get("siblings", [])
    filenames = {s["rfilename"] for s in siblings}

    config = None
    if "config.json" in filenames:
        config_url = f"{_get_hf_endpoint()}/{repo_id}/resolve/main/config.json"
        config = json.loads(_make_request(config_url, timeout=timeout).decode("utf-8"))

    tensors = {}
    total_size = 0.0

    if "model.safetensors.index.json" in filenames:
        # Sharded SafeTensors
        index_url = f"{_get_hf_endpoint()}/{repo_id}/resolve/main/model.safetensors.index.json"
        index_data = json.loads(
            _make_request(index_url, timeout=timeout).decode("utf-8")
        )

        weight_map = index_data.get("weight_map", {})
        unique_shards = list(set(weight_map.values()))

        total_size = index_data.get("metadata", {}).get("total_size", 0.0)

        if config and not fetch_tensors and total_size > 0:
            # Lazy Fetch Paradigm
            for tensor_name in weight_map.keys():
                tensors[tensor_name] = {"shape": [], "dtype": "BF16"}

            tensors["__metadata__"] = {
                "missing_shards": 0,
                "total_shards": len(unique_shards),
                "is_sharded": True,
                "lazy_fetch": True,
                "total_size": total_size,
            }
        else:

            def fetch_shard(shard: str):
                return shard, _fetch_safetensors_header(repo_id, shard, timeout=timeout)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(8, len(unique_shards)))
            ) as executor:
                future_to_shard = {
                    executor.submit(fetch_shard, shard): shard
                    for shard in unique_shards
                }
                for future in concurrent.futures.as_completed(future_to_shard):
                    shard, shard_header = future.result()
                    for k, v in shard_header.items():
                        if k != "__metadata__":
                            tensors[k] = v

            tensors["__metadata__"] = {
                "missing_shards": 0,
                "total_shards": len(unique_shards),
                "is_sharded": True,
            }
        format_name = "SafeTensors"

    elif "model.safetensors" in filenames:
        # Single SafeTensors

        # Determine total size first
        req = urllib.request.Request(
            f"{_get_hf_endpoint()}/{repo_id}/resolve/main/model.safetensors",
            method="HEAD",
        )
        token = _get_hf_token()
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("Content-Length", 0))
        except Exception:
            pass

        header = _fetch_safetensors_header(
            repo_id, "model.safetensors", timeout=timeout
        )
        tensors = header

        format_name = "SafeTensors"

    else:
        raise ValueError(f"Repository {repo_id} does not contain SafeTensors weights.")

    return tensors, config, format_name, float(total_size)
