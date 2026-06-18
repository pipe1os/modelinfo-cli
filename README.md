# ModelInfo CLI

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-rich-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

![ModelInfo Demo](modelinfo.gif)

ModelInfo is a CLI tool that inspects machine learning model checkpoints (`.safetensors`, `.gguf`, `.pt`) and calculates hardware requirements completely offline.

It reads binary headers directly using the Python standard library. It skips the full tensor payload entirely (no PyTorch, no HuggingFace) and parses in under 100ms.

## Features

- **Zero-Dependency Parsing**: Reads `.safetensors` 8-byte JSON prefixes and `.gguf` binary key-value metadata directly via `struct` and `json` (falling back to `config.json` if needed).
- **Remote Hugging Face Hub Inspection**: Pass a repo ID (e.g., `meta-llama/Llama-2-7b-hf`) and it uses concurrent byte-range requests to read the headers off the CDN in under 2 seconds. No need to download the checkpoint.
- Parses `model.safetensors.index.json` to support sharded models without crashing on partial downloads.
- **Dynamic VRAM & vLLM Capacity Planning**: Calculates exact VRAM limits based on the model's architecture and your target context length. If you use the `--vllm` flag, it switches to a "Serving Capacity" simulation that calculates exactly how many tokens fit in the PagedAttention pool based on your `--gpu-util` ratio.
- **Hardware Fit Diagnostics**: Check if a model fits your cluster with `--gpu` (e.g. `--gpu RTX4090` or `--gpu auto`). It enforces Apple Silicon's 75% unified memory wire limit, and you can explicitly model multi-GPU NCCL communication penalties with `--topology` and `--strategy`.
- **Side-by-Side Comparison**: Pass multiple models to trigger a comparison table (parameters, data types, context lengths, VRAM footprints).
- Uses exact `ggml_type` mappings for GGUF formats to calculate byte-scaling coefficients, preventing VRAM under-reporting.
- **Secure Pickling**: Inspects legacy `.pt` files safely using a restricted `pickle.Unpickler`.
- The UI (built with `rich`) groups repetitive layers and color-codes VRAM heatmaps.

> [!NOTE]
> **A Note on Performance & Remote Fetching**
> Local `.gguf` and `.safetensors` files are parsed in under 100ms. However, querying remote Hugging Face repositories takes **1 to 10 seconds**. To remain zero-dependency, `modelinfo` opens connections via Python `urllib` instead of loading PyTorch. For massive sharded models (e.g., 100+ shards), it must fetch every header individually, capped at an 8-worker thread pool to prevent Cloudflare IP bans. Waiting ~8 seconds to map a model is faster than downloading 400GB just to see if it fits your hardware.

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

Tests cover the binary parsers and verify the sub-100ms local parse constraint using binary mocks in `tests/fixtures/`.

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

Simulate exactly how many tokens you can serve using vLLM on a specific multi-GPU topology:

```bash
modelinfo mistralai/Mistral-7B-v0.1 --vllm --gpu 4xRTX4090 --topology pcie4 --strategy tp
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
| `--batch-size` | `--batch-size 32` | Batch size for dynamic KV cache footprint calculation. Defaults to `1`. |
| `--max-vram` | `--max-vram 80` | Adjusts the color-coded heat mapping thresholds (Green/Yellow/Red) in the terminal output to match a specific hardware ceiling. |
| `--vllm` | `--vllm --gpu auto` | Switches from additive memory checking to a serving capacity simulation. Shows exactly how many tokens fit in the PagedAttention pool. |
| `--gpu-util` | `--gpu-util 0.9` | Sets the vLLM `gpu_memory_utilization` ratio. Defaults to `0.9` (reserves 10% for PyTorch context). |
| `--topology` | `--topology nvlink` | Set interconnect topology to calculate exact communication overhead penalties (`nvlink`, `pcie4`, `pcie3`). Defaults to `pcie4`. |
| `--strategy` | `--strategy tp` | Selects the parallelization strategy for multi-GPU setups (`tp` for Tensor Parallelism, `pp` for Pipeline Parallelism). Defaults to `tp`. |
| `--tensors` | `--tensors` | Bypasses the algorithmic speed estimation and forces the tool to fetch all remote shards, displaying an exact size breakdown of every tensor. |
| `--timeout` | `--timeout 30` | Network request timeout in seconds for Hugging Face Hub. Defaults to `10.0`. |
| `-v, --version` | `modelinfo -v` | Show program's version number and exit. |

## Architecture

Three modules:

1. **Presentation (`cli.py`, `ui.py`)**: Parses arguments and formats tables via `rich`.
2. **Parsing Engine (`parsers/`)**: Specialized binary readers (`safetensors.py`, `gguf.py`, `pytorch.py`) that use only the standard library.
3. **Math Engine (`calculator.py`)**: Determines total parameter counts, maps data types to byte coefficients, and calculates dynamic memory allocations based on tensor shape heuristics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
