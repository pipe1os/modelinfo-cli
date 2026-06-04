# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased]
 
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
