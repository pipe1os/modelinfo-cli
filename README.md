# ModelInfo CLI

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-rich-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

ModelInfo is a terminal-native utility that inspects machine learning model checkpoints (`.safetensors`, `.gguf`, `.pt`) and calculates hardware requirements completely offline.

It reads binary headers directly using the Python standard library. By bypassing full tensor payload loading and strictly excluding heavy ecosystems like PyTorch or HuggingFace, the tool executes in under 100 milliseconds.

## Features

- **Zero-Dependency Parsing**: Reads the 8-byte JSON prefix of `.safetensors` files and the binary key-value metadata of `.gguf` directly via `struct` and `json`. Reads adjacent `config.json` for architecture fallback.
- **Remote Hugging Face Hub Inspection**: Inspect any public or gated model directly via its repo ID (e.g., `modelinfo meta-llama/Llama-2-7b-hf`) without downloading the checkpoint. Uses concurrent byte-range requests to read the binary headers directly off the CDN in under 2 seconds.
- **Sharded Model Support**: Transparently parses `model.safetensors.index.json` to detect multi-file checkpoint distributions, gracefully guarding against partial downloads without crashing.
- **Dynamic VRAM Estimation**: Extracts underlying model architecture to calculate exact VRAM limits, including dynamic KV cache footprints based on user-specified context lengths.
- **Hardware Fit Diagnostics**: Pass the `--gpu` flag (e.g. `--gpu RTX4090` or `--gpu auto`) to calculate if the model fits in your specific cluster. Defends against fragmentation OOMs using a 3-tier heuristic (Safe, Warning, Fail), calculates overhead across multi-GPU setups, and enforces Apple Silicon's 75% unified memory wire limit.
- **Side-by-Side Comparison**: Pass multiple models to automatically trigger a comparison table. Compares parameters, data types, context lengths, and VRAM footprints side-by-side to evaluate trade-offs.
- **Precise Block Quantization**: Factors in exact byte-scaling coefficients for GGUF formats (e.g., Q8, Q6, Q4) rather than naive averages, eliminating VRAM under-reporting.
- **Secure Pickling**: Inspects legacy `.pt` files without executing arbitrary code by using a highly restricted `pickle.Unpickler`.
- **Terminal UI**: Groups repetitive structural layers and color-codes VRAM heatmaps using `rich`. Breaks down memory footprints into Weights, KV Cache, and Overhead.

> [!NOTE]
> **A Note on Performance & Remote Fetching**
> Local `.gguf` and `.safetensors` files are parsed in under 100ms. However, querying remote Hugging Face repositories takes **1 to 10 seconds**. This is an intentional trade-off. To remain zero-dependency, `modelinfo` negotiates raw TCP/TLS via Python `urllib` instead of loading PyTorch. For massive sharded models (e.g., 100+ shards), it must fetch every header individually, capped at an 8-worker thread pool to prevent Cloudflare IP bans. Waiting ~8 seconds to map a model is faster than downloading 400GB just to see if it fits your hardware.

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

For gated models (e.g., Llama 2), you must provide authentication by setting the `HF_TOKEN` environment variable. You can create a token in your [Hugging Face settings](https://huggingface.co/settings/tokens).

```bash
export HF_TOKEN="hf_your_token_here"
modelinfo meta-llama/Llama-2-7b-hf
```

Alternatively, the tool will automatically read tokens stored by the `hf auth login` command (located in `~/.cache/huggingface/token`).

Calculate the memory footprint with a specific KV cache context window:

```bash
modelinfo mistral-7b.safetensors --context 8192
```

Adjust the VRAM heat-mapping thresholds for your specific hardware (e.g., an 80GB card):

```bash
modelinfo meta-llama/Llama-2-7b-hf --max-vram 80
```

Determine if a model fits your specific hardware:

```bash
modelinfo mistralai/Mistral-7B-v0.1 --gpu "RTX 4090"
modelinfo mistralai/Mistral-7B-v0.1 --gpu auto
```

Compare multiple models side-by-side against a hardware target:

```bash
modelinfo mistralai/Mistral-7B-v0.1 Qwen/Qwen2.5-0.5B --gpu 12
```

### Example Output (Single Model)

```text
Format:          SafeTensors
Architecture:    MistralForCausalLM (32 layers)
Tensors:         291
Parameters:      7.2B
Dtype:           BF16
Disk size:       13.49 GB
VRAM (est):      ~15.07 GB Total Minimum Required
                   ├─ Weights:    13.49 GB
                   ├─ KV Cache:   1.0 GB (Default 8192 tokens. Native limit: 32,768)
                   └─ Overhead:   600.0 MB (CUDA Context + Activations)
Hardware Fit:    ✗ No (Requires 15.07 GB, Hardware has 12.0 GB)

Top Tensors by Size:
  model.embed_tokens.weight                     [32000 x 4096]   bf16   131.1M params
  32x model.layers.[N].self_attn.q_proj.weight  [4096 x 4096]    bf16    16.8M params
```

### Example Output (Comparison)

```text
Model              Params    Dtype    Context    VRAM        Fits
Mistral-7B-v0.1    7.2B      BF16     8K         15.07 GB    ✗
Qwen2.5-0.5B       494.0M    BF16     8K         1.6 GB      ✓
```

## Command Reference

| Argument | Example | Description |
| :--- | :--- | :--- |
| `[files...]` | `modelinfo model.safetensors` | Inspect a single model (local path or Hugging Face repo ID). |
| `[files...]` | `modelinfo modelA modelB` | Pass multiple files/repos to automatically render a side-by-side comparison table instead of a deep-dive summary. |
| `--gpu` | `--gpu rtx4090` | Check if the model fits. Accepts GPU names (`rtx4090`, `b200`, `rx7900xtx`), explicit VRAM limits in GB (`--gpu 24`), or local hardware auto-discovery (`--gpu auto`). |
| `--context` | `--context 32768` | Adjust the target KV cache length. Essential for calculating the dynamic memory footprint of long-context models. Defaults to `8192`. |
| `--max-vram` | `--max-vram 80` | Adjusts the color-coded heat mapping thresholds (Green/Yellow/Red) in the terminal output to match a specific hardware ceiling. |

## Architecture

The system operates across three modules:

1. **Presentation (`cli.py`, `ui.py`)**: Parses arguments and formats tables via `rich`.
2. **Parsing Engine (`parsers/`)**: Specialized binary readers (`safetensors.py`, `gguf.py`, `pytorch.py`) strictly confined to standard library operations.
3. **Math Engine (`calculator.py`)**: Determines total parameter counts, maps data types to byte coefficients, and calculates dynamic memory allocations based on tensor shape heuristics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
