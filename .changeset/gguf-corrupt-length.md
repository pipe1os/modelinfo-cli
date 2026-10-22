---
"modelinfo-cli": patch
---

Fixed a `MemoryError` on GGUF files with a corrupt string or array length; they now fail with "Unexpected end of GGUF header".
