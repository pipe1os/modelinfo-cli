# ModelInfo CLI

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-rich-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

ModelInfo is a terminal-native utility that inspects machine learning model checkpoints (`.safetensors`, `.gguf`, `.pt`) and calculates hardware requirements completely offline.

It reads binary headers directly using the Python standard library. By bypassing full tensor payload loading and strictly excluding heavy ecosystems like PyTorch or HuggingFace, the tool executes in under 100 milliseconds.

## Features

- **Zero-Dependency Parsing**: Reads the 8-byte JSON prefix of `.safetensors` files and the binary key-value metadata of `.gguf` directly via `struct` and `json`. Seamlessly reads adjacent `config.json` for robust fallback logic.
- **Remote Hugging Face Hub Inspection**: Inspect any public or gated model directly via its repo ID (e.g., `modelinfo meta-llama/Llama-2-7b-hf`) without downloading the 15GB checkpoint. Uses concurrent byte-range requests to pluck the binary headers directly off the CDN in under 2 seconds.
- **Sharded Model Support**: Transparently parses `model.safetensors.index.json` to detect multi-file checkpoint distributions, gracefully guarding against partial downloads without crashing.
- **Dynamic VRAM Estimation**: Extracts underlying model architecture (layers, heads, dimensions) to calculate exact VRAM limits, including dynamic KV cache footprints based on user-specified context lengths. Defaults to 8192 tokens to prevent unrealistic VRAM calculations, while still warning users if the requested context exceeds the model's native limit. Estimates include a standard 600MB CUDA context overhead.
- **Precise Block Quantization**: Factors in exact byte-scaling coefficients for GGUF formats (e.g., Q8, Q6, Q4) rather than naive averages, eliminating VRAM under-reporting.
- **Secure Pickling**: Inspects legacy `.pt` files without executing arbitrary code by using a highly restricted `pickle.Unpickler`.
- **Terminal UI**: Groups repetitive structural layers and color-codes VRAM heatmaps using `rich`. Breaks down memory footprints into Weights, KV Cache, and Overhead.

## Installation

Install directly from PyPI:

```bash
pip install modelinfo-cli
```

### Development

To install from source and run the test suite:

```bash
git clone https://github.com/pipe1os/modelinfo-cli.git
cd modelinfo-cli
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing

The testing suite enforces cross-platform structural integrity and guards the zero-dependency latency constraint. Tests are isolated against custom binary mocks in `tests/fixtures/`.

Run the test suite using pytest:

```bash
pytest tests/ -v
```

## Usage

Inspect a local model checkpoint:

```bash
modelinfo mistral-7b.safetensors
```

Inspect a remote model directly from the Hugging Face Hub:

```bash
modelinfo meta-llama/Llama-2-7b-hf
```

Calculate the memory footprint with a specific KV cache context window:

```bash
modelinfo mistral-7b.safetensors --context 8192
```

### Example Output

```text
Format:         SafeTensors
Architecture:   Mistral (32 transformer layers)
Tensors:        291
Parameters:     7.2B
Dtype:          bf16
Disk size:      13.49 GB
VRAM (est):     ~15.2 GB (bf16, KV cache for 8192 tokens)

Top Tensors by Size:
  model.embed_tokens.weight                     [32000 x 4096]   bf16   131.1M params
  32x model.layers.[N].self_attn.q_proj.weight  [4096 x 4096]    bf16    16.8M params
```

## Architecture

The system operates across three modules:

1. **Presentation (`cli.py`, `ui.py`)**: Parses arguments and formats tables via `rich`.
2. **Parsing Engine (`parsers/`)**: Specialized binary readers (`safetensors.py`, `gguf.py`, `pytorch.py`) strictly confined to standard library operations.
3. **Math Engine (`calculator.py`)**: Determines total parameter counts, maps data types to byte coefficients, and calculates dynamic memory allocations based on tensor shape heuristics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
