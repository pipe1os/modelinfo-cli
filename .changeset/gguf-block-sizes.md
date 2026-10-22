---
"modelinfo-cli": patch
---

Fixed GGUF quantized memory estimates to match the GGML block sizes (e.g. Q4_K was overestimated by ~5.6%) and counted scalar tensors (#52).
