---
"modelinfo-cli": patch
---

Fixed PyTorch `.pt`/`.bin` parsing failing on every real `torch.save` checkpoint, and added support for double, short, char, byte and bool storage and wrapped `state_dict` checkpoints (#45).
