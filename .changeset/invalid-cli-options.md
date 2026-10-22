---
"modelinfo-cli": patch
---

Rejected invalid `--context`, `--max-vram`, `--gpu-util` and `--gpu` values (zero GPUs, nonpositive or non-finite VRAM) instead of producing negative or NaN estimates (#49, #51).
