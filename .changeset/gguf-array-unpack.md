---
"modelinfo-cli": patch
---

GGUF headers parse about twice as fast: numeric metadata arrays such as tokenizer scores and token types are unpacked in one call.
