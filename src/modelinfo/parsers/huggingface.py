import concurrent.futures
import json
import os
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


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


def _hub_file_url(repo_id: str, filename: str) -> str:
    repo_path = urllib.parse.quote(repo_id, safe="/")
    file_path = urllib.parse.quote(filename, safe="/")
    return f"{_get_hf_endpoint()}/{repo_path}/resolve/main/{file_path}"


def _get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
        
    cache_home = os.environ.get("XDG_CACHE_HOME", "~/.cache")
    hf_home = os.environ.get("HF_HOME", os.path.join(cache_home, "huggingface"))
    cache_path = os.path.expanduser(os.environ.get("HF_TOKEN_PATH", os.path.join(hf_home, "token")))
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
    headers: dict[str, str] | None = None,
    limit: int | None = None,
    timeout: float = 10.0,
) -> bytes:
    headers = dict(headers) if headers is not None else {}
        
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            requested_range = req.get_header("Range", "")
            if (requested_range.startswith("bytes=")
                    and requested_range.split("=", 1)[1].split("-", 1)[0] != "0"
                    and getattr(response, "status", None) == 200):
                raise ValueError("Server ignored the requested byte range; refusing data from the wrong offset.")
            content_range = getattr(response, "headers", {}).get("Content-Range")
            if requested_range.startswith("bytes=") and getattr(response, "status", None) == 206 and content_range:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", content_range)
                expected_start = int(requested_range.split("=", 1)[1].split("-", 1)[0])
                if (match is None or int(match[1]) != expected_start
                        or int(match[2]) < int(match[1])):
                    raise ValueError("Server returned data for an invalid or unexpected byte range.")
            if limit is not None:
                return response.read(limit)
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(f"Gated/Private Model or Invalid Token (401 Unauthorized). Set the HF_TOKEN environment variable to access {url}")
        if e.code == 404:
           raise FileNotFoundError(f"Could not find repository or file on Hugging Face (404 Not Found): {url}")
        raise

def _fetch_safetensors_header(repo_id: str, filename: str, timeout: float = 10.0) -> dict[str, Any]:
    url = _hub_file_url(repo_id, filename)
    
    # 1. Fetch the first 500KB in a single roundtrip
    headers = {"Range": "bytes=0-500000"}
    try:
        chunk = _make_request(url, headers=headers, limit=500000, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416: # Range Not Satisfiable (file is smaller than 500KB)
            chunk = _make_request(url, limit=500000, timeout=timeout)
        else:
            raise
            
    if len(chunk) < 8:
        raise ValueError(f"File {filename} is too small to contain a SafeTensors header.")
        
    header_size = struct.unpack("<Q", chunk[:8])[0]
    if header_size > 100 * 1024 * 1024:
        raise ValueError(f"Header length ({header_size} bytes) exceeds maximum allowed size.")
    
    # 2. Slice locally if it fits
    if 8 + header_size <= len(chunk):
        json_bytes = chunk[8:8+header_size]
    else:
        # 3. Double-roundtrip only if the header is massive (>500KB)
        headers = {"Range": f"bytes=8-{8+header_size-1}"}
        json_bytes = _make_request(url, headers=headers, limit=header_size, timeout=timeout)
        
    if len(json_bytes) != header_size:
        raise ValueError(f"SafeTensors header in {filename} is truncated: expected {header_size} bytes, got {len(json_bytes)}.")

    header = json.loads(json_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"SafeTensors header in {filename} must be a JSON object.")  # noqa: TRY004 - ValueError is the documented error for bad headers
    return header

def _get_remote_file_size_fallback(repo_id: str, filename: str, timeout: float = 10.0) -> float:
    req = urllib.request.Request(_hub_file_url(repo_id, filename), method="HEAD")
    token = _get_hf_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return float(response.headers.get("Content-Length", 0))
    except Exception:  # noqa: BLE001 - size is optional, 0.0 means unknown
        return 0.0


class RemoteFileStream:
    def __init__(self, url: str, chunk_size: int = 1024*1024, timeout: float = 10.0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.url = url
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.buffer = bytearray()
        self.position = 0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size == -1:
            raise NotImplementedError("Unlimited remote read is not supported.")
            
        if size < -1:
            raise ValueError("read size must be nonnegative or -1")
        end_pos = self.position + size
        if end_pos > 50 * 1024 * 1024:
            raise ValueError("Remote header read limit exceeded (50MB). File might be invalid or too large.")
            
        while end_pos > len(self.buffer):
            start_bytes = len(self.buffer)
            end_bytes = start_bytes + self.chunk_size - 1
            
            headers = {"Range": f"bytes={start_bytes}-{end_bytes}"}
            try:
                chunk = _make_request(
                    self.url,
                    headers=headers,
                    limit=self.chunk_size,
                    timeout=self.timeout
                )
                if not chunk:
                    break
                self.buffer.extend(chunk)
            except urllib.error.HTTPError as e:
                if e.code == 416:
                    break
                raise
            except Exception:
                raise
                
        result = bytes(self.buffer[self.position:self.position+size])
        self.position += len(result)
        return result

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        else:
            raise NotImplementedError("Seek from end is not supported.")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return self.position

    def tell(self) -> int:
        return self.position

    def close(self) -> None:
        pass


def _fetch_remote_gguf_single(real_repo_id: str, filename: str, fallback_size: float | None, timeout: float) -> tuple[dict[str, Any], float]:
    url = _hub_file_url(real_repo_id, filename)
    stream = RemoteFileStream(url, timeout=timeout)
    from modelinfo.parsers.gguf import parse_gguf_header
    tensors = parse_gguf_header(stream)
    
    size = fallback_size if fallback_size is not None else 0.0
    if size == 0.0:
        size = _get_remote_file_size_fallback(real_repo_id, filename, timeout)
    return tensors, size


def _fetch_remote_gguf_group(real_repo_id: str, gguf_files: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    valid_sizes = [g for g in gguf_files if g["size"] is not None and g["size"] > 0]
    if valid_sizes:
        header_target = min(valid_sizes, key=lambda x: x["size"])
    else:
        header_target = gguf_files[0]
    
    header_file = header_target["filename"]
    url = _hub_file_url(real_repo_id, header_file)
    stream = RemoteFileStream(url, timeout=timeout)
    from modelinfo.parsers.gguf import parse_gguf_header
    tensors = parse_gguf_header(stream)
    
    variants = []
    for g in gguf_files:
        v_size = g["size"]
        if v_size is None or v_size == 0:
            v_size = _get_remote_file_size_fallback(real_repo_id, g["filename"], timeout)
        variants.append({
            "filename": g["filename"],
            "size": float(v_size)
        })
    
    tensors["__metadata__"] = tensors.get("__metadata__", {})
    tensors["__metadata__"]["gguf_variants"] = variants
    tensors["__metadata__"]["repo_id"] = real_repo_id
    return tensors


def _fetch_shards_concurrently(real_repo_id: str, unique_shards: list[str], timeout: float, weight_map: dict[str, str] | None = None) -> tuple[dict[str, Any], int]:
    def fetch_shard(shard: str):
        try:
            header = _fetch_safetensors_header(real_repo_id, shard, timeout=timeout)
            if weight_map is not None:
                assigned = [name for name, filename in weight_map.items() if filename == shard]
                if any(name not in header for name in assigned):
                    raise ValueError(f"Indexed tensor missing from shard {shard!r}")
                header = {name: header[name] for name in assigned}
            return shard, header, None
        except Exception as e:  # noqa: BLE001 - any shard failure is reported to the caller
            return shard, {}, e
        
    tensors = {}
    missing_shards = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(unique_shards)))) as executor:
        future_to_shard = {executor.submit(fetch_shard, shard): shard for shard in unique_shards}
        for future in concurrent.futures.as_completed(future_to_shard):
            _shard, shard_header, error = future.result()
            if error is not None:
                missing_shards += 1
            else:
                for k, v in shard_header.items():
                    if k != "__metadata__":
                        tensors[k] = v
    return tensors, missing_shards


def _fetch_remote_safetensors_sharded(
    real_repo_id: str,
    config: dict[str, Any] | None,
    fetch_tensors: bool,
    timeout: float
) -> tuple[dict[str, Any], float]:
    index_url = _hub_file_url(real_repo_id, "model.safetensors.index.json")
    index_data = json.loads(_make_request(index_url, timeout=timeout).decode("utf-8"))
    
    weight_map = index_data.get("weight_map", {})
    unique_shards = list(set(weight_map.values()))
    total_size = index_data.get("metadata", {}).get("total_size", 0.0)
    
    tensors = {}
    if config and not fetch_tensors and total_size > 0:
        for tensor_name in weight_map:
            tensors[tensor_name] = {"shape": [], "dtype": "BF16"}
            
        tensors["__metadata__"] = {
            "missing_shards": 0,
            "total_shards": len(unique_shards),
            "is_sharded": True,
            "lazy_fetch": True,
            "total_size": total_size
        }
    else:
        tensors, missing_shards = _fetch_shards_concurrently(real_repo_id, unique_shards, timeout, weight_map)
        tensors["__metadata__"] = {
            "missing_shards": missing_shards,
            "total_shards": len(unique_shards),
            "is_sharded": True
        }
    return tensors, float(total_size)



def _fetch_remote_safetensors_single(real_repo_id: str, timeout: float) -> tuple[dict[str, Any], float]:
    total_size = 0.0
    req = urllib.request.Request(_hub_file_url(real_repo_id, "model.safetensors"), method="HEAD")
    token = _get_hf_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", 0))
    except Exception:  # noqa: BLE001, S110 - size is optional, keep default
        pass

    header = _fetch_safetensors_header(real_repo_id, "model.safetensors", timeout=timeout)
    return header, float(total_size)


def fetch_huggingface_repo(repo_id: str, fetch_tensors: bool = False, timeout: float = 10.0) -> tuple[dict[str, Any], dict[str, Any] | None, str, float]:
    """
    Fetches the metadata directly from the Hugging Face Hub over the network.
    Returns: (tensors, config, format_name, disk_size)
    """
    target_filename = None
    parts = repo_id.split("/")
    if len(parts) >= 3 and parts[-1].lower().endswith(".gguf"):
        real_repo_id = "/".join(parts[:2])
        target_filename = "/".join(parts[2:])
    else:
        real_repo_id = repo_id

    api_url = f"{_get_hf_endpoint()}/api/models/{real_repo_id}"
    try:
        api_data = json.loads(_make_request(api_url, timeout=timeout).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise PermissionError(f"Gated/Private Model (401 Unauthorized). Set the HF_TOKEN environment variable to access {real_repo_id}")
        if e.code == 404:
             raise FileNotFoundError(f"Could not find repository on Hugging Face (404 Not Found): {real_repo_id}")
        raise
        
    siblings = api_data.get("siblings", [])
    filenames = {s["rfilename"] for s in siblings}
    
    config = None
    if "config.json" in filenames:
        config_url = _hub_file_url(real_repo_id, "config.json")
        config = json.loads(_make_request(config_url, timeout=timeout).decode("utf-8"))

    # Find GGUF siblings
    gguf_files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if fname.lower().endswith(".gguf"):
            gguf_files.append({
                "filename": fname,
                "size": s.get("size")
            })

    if target_filename:
        target_sibling = next((g for g in gguf_files if g["filename"] == target_filename), None)
        if not target_sibling:
            raise FileNotFoundError(f"Could not find file '{target_filename}' in Hugging Face repository '{real_repo_id}'.")
        tensors, size = _fetch_remote_gguf_single(real_repo_id, target_filename, target_sibling["size"], timeout)
        return tensors, config, "GGUF", float(size)

    # Fallback to SafeTensors checks if no specific file is target
    if "model.safetensors.index.json" in filenames:
        tensors, total_size = _fetch_remote_safetensors_sharded(real_repo_id, config, fetch_tensors, timeout)
        return tensors, config, "SafeTensors", total_size
        
    elif "model.safetensors" in filenames:
        header, total_size = _fetch_remote_safetensors_single(real_repo_id, timeout)
        return header, config, "SafeTensors", total_size

    elif gguf_files:
        if len(gguf_files) == 1:
            single_file = gguf_files[0]["filename"]
            tensors, size = _fetch_remote_gguf_single(real_repo_id, single_file, gguf_files[0]["size"], timeout)
            return tensors, config, "GGUF", float(size)
        else:
            tensors = _fetch_remote_gguf_group(real_repo_id, gguf_files, timeout)
            return tensors, config, "GGUF_group", 0.0

    else:
        raise ValueError(f"Repository {real_repo_id} does not contain SafeTensors or GGUF weights.")

