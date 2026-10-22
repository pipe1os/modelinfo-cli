---
"modelinfo-cli": patch
---

Fixed an `UnboundLocalError` when `--gpu` was combined with missing shards or missing tensor shapes (#47).
