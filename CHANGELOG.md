# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased]

## [1.4.4] - 2026-06-27

### Added
- Added the `--batch-size` flag (defaulting to 1) for dynamic KV cache footprint calculations.
- Added the `--timeout` flag (defaulting to 10s) to configure network timeouts for remote Hugging Face fetches.
- Added support for custom Hugging Face endpoints via the `HF_ENDPOINT` environment variable.
- Added auto-discovery and memory capacity mapping for Intel GPUs using `xpu-smi`.
- Added suggestions for similar GPU names using `difflib.get_close_matches` when an unrecognized GPU target is provided.

### Changed
- Reorganized local GPU discovery helpers in `hardware.py`.
- Cleaned up test parser module imports to resolve E402 warnings.

### Fixed
- Propagated timeout values to all remote fetch requests in the Hugging Face parser.

## [1.4.3] - 2026-06-13

### Added
- Added the `-v`/`--version` flag to quickly check the installed modelinfo version. The version lookup is lazily evaluated to guarantee sub-100ms CLI startup times.
- Added missing entry-level GPUs to the `KNOWN_GPUS` hardware discovery dictionary.
- Added repository documentation including `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and GitHub issue/PR templates.

### Fixed
- Fixed an Out-Of-Memory (OOM) vulnerability during remote inspection by capping the HTTP response read. This protects the CLI from upstream CDN proxies that ignore HTTP `Range` headers.
- Fixed confusing stack traces when a local directory is passed instead of a file by raising an explicit `IsADirectoryError`.
- Fixed the CLI to print user-friendly error messages when attempting to inspect gated or non-existent Hugging Face repositories (401 Unauthorized / 404 Not Found).
- Fixed an issue where the main entry point swallowed exceptions too broadly, obscuring critical stack traces during unexpected failures.

## [1.4.2] - 2026-06-08

### Fixed
- Fixed unused imports (`os`, `json`) in architecture parsing logic.
- Fixed a bug where the `--max-vram` argument was ignored when evaluating single models without a target GPU.
- Fixed a bug where the target GPU's memory limit was ignored in favor of the default max VRAM when rendering the multi-model comparison table.
- Prevented a potential `ValueError` crash in the remote fetcher by enforcing a minimum of 1 worker for the `ThreadPoolExecutor`.

## [1.4.1] - 2026-06-07

### Changed
- Updated repository documentation to reflect precise terminology for new hardware features.

## [1.4.0] - 2026-06-07

This release adds multi-GPU hardware topology modeling and a vLLM serving capacity engine for inference planning. We also overhauled how remote Hub interactions work to speed up metadata fetching.

### Added
- Added the `--vllm` flag to switch from additive VRAM checks to a "Serving Capacity" simulation. It calculates PagedAttention block limits based on a configurable `--gpu-util` ratio.
- **Topology-Aware Overhead Scaling:** Added `--topology` (`nvlink`, `pcie4`, `pcie3`) and `--strategy` (`tp`, `pp`) flags. The calculator now applies NCCL communication penalties directly to weights and activations instead of using a generic fixed multiplier.
- Mapped explicit `ggml_type` enums (0-33) for GGUF files to fix VRAM under-reporting for specific quantization types.
- The CLI now does algorithmic estimation via `index.json` by default. If you need the exact size breakdown of every tensor, pass `--tensors` to force it to fetch all remote shards.
- Added comprehensive `pytest` test coverage for the new vLLM serving capacity engine, topology penalties, and explicit GGUF byte mappings.

### Changed
- Removed KV cache from the distributed overhead multiplier because Tensor Parallelism partitions context blocks rather than duplicating them.
- Changed the network logic to infer metadata directly from `index.json` and `config.json`. It skips iterative chunk requests for sharded arrays unless `--tensors` is passed.
## [1.3.0] - 2026-06-06

This release adds comprehensive hardware fit diagnostics, dynamic GPU scaling, and side-by-side model comparison to instantly evaluate operational deployment trade-offs.

### Added
- **Hardware Discovery Engine:** Added the `--gpu` flag with multi-vendor normalization (NVIDIA, AMD, Intel) to calculate if a model fits within specific hardware constraints. Supports named GPUs (`rtx4090`), explicit sizes (`24`), and native local hardware discovery (`auto`).
- **Fragmentation Defense:** Implemented a 3-tier UI heuristic (✓ Safe, ⚠ Warning, ✗ Fail) to defend against memory fragmentation and generation-time transient spikes.
- **Side-by-Side Comparison:** Passing multiple models via the CLI (`modelinfo modelA modelB`) now implicitly triggers a dedicated side-by-side comparison table, surfacing parameter counts, context lengths, and VRAM footprints to evaluate architectural trade-offs.

### Changed
- **Multi-GPU Overhead Scaling:** The CUDA context initialization overhead (600 MB) now dynamically scales based on the detected `gpu_count` to prevent silent prefill OOMs on multi-GPU deployments.
- **Mathematical Transparency:** Enforced the `Dtype` column mathematically into the comparative UI to visualize exactly why quantization scales VRAM footprints downward.

## [1.2.0] - 2026-06-04

This release adds remote Hugging Face Hub inspection, dynamic VRAM overhead modeling, and sensible context defaults for operational inference planning.

### Added
- **Remote Hugging Face Hub Support:** Inspect any public or gated model directly via its repo ID (e.g., `modelinfo meta-llama/Llama-2-7b-hf`) without downloading the full checkpoint. Uses concurrent `Range` requests (max 8 workers) to extract the first 500KB of safetensors shards to prevent synchronous I/O bottlenecks and bypass CDN rate-limits.
- **Framework Overhead Modeling:** VRAM estimates now include a static 600 MB CUDA context overhead alongside the model weights and KV cache for operational accuracy.
- **Hierarchical VRAM UI:** Redesigned the output terminal UI to group memory footprints into Weights, KV Cache, and Overhead.

### Changed
- **Sane Context Defaults:** Hard-capped the default `--context` value at 8192 tokens. Models with extreme architectural boundaries (e.g., 128k) will still read the native limit and print it in the UI, protecting users from unrealistic default memory calculations.
- **Authentication Fallback:** Remote HTTP fetcher now supports token extraction from the `HF_TOKEN` environment variable, `~/.cache/huggingface/token`, and the legacy `~/.huggingface/token` path.

## [1.1.0] - 2026-06-04
 
This release introduces deterministic architecture parsing and native context limit warnings, shifting the tool from heuristic guesswork to explicit binary metadata extraction.
 
### Added
- **GGUF Metadata Extraction:** The parser now extracts global key-value pairs (e.g., `general.architecture`, `attention.head_count_kv`) to guarantee accurate architecture mapping.
- **Context Limit Warnings:** Extracts `max_position_embeddings` (Hugging Face) and `{gen_arch}.context_length` (GGUF) to actively warn users if requested `--context` exceeds the model's native boundary.
- **SafeTensors Config Fallback:** Seamlessly reads adjacent `config.json` files for robust fallback parsing of architectures lacking explicit tensor structures.
 
### Changed
- **Decoupled Math Engine:** Moved all filesystem operations and config parsing into `cli.py` to maintain Separation of Concerns, keeping `calculator.py` and `architecture.py` as pure, testable math engines.
 
### Fixed
- **Fused Tensor Estimation:** Defused mathematical edge cases for GQA, ALiBi, and older MHA models when extracting KV cache dimensionality from fused `qkv_proj.weight` tensors.
- **Terminal UI Layout:** Solved terminal text-wrapping bugs by constraining `rich` table columns and removing non-standard emojis for a professional, clean Unix aesthetic.

## [1.0.0] - 2026-06-02
 
Initial public release of the `modelinfo-cli` package.
 
### Added
- **Zero-Dependency Parsers**: Native standard-library (`os`, `struct`, `json`) binary deserialization for `.safetensors`, `.gguf`, and legacy `.pt` files.
- **Sharded Architecture Support**: Auto-detects and gracefully parses `model.safetensors.index.json` manifests without crashing on partial downloads.
- **Hardware Calculator**: Calculates exact VRAM footprints (including dynamic KV cache overhead) using precise GGUF block quantization ratios (Q8 through Q2).
- **Restricted Unpickler**: Defangs arbitrary code execution when inspecting legacy PyTorch checkpoints.
