# Weights

Curated milestone checkpoints — enough to reproduce every curve in
`results/`. Full 1k-granularity sets are GitHub Release assets (clones stay
light).

```
run100k/   v2, the 1255-Elo baseline — ckpt_010000 ... ckpt_100000 (every 10k)
```

## Format

Each `.pt` is a config-aware checkpoint: `{"config": {...}, "state": {...}}`.
Load with the helper, which also upgrades older checkpoints (18-feature
input, scalar value head) to the current architecture in place:

```python
import torch  # before numpy — see ../requirements.txt
from prophet.model import load_checkpoint

model = load_checkpoint("weights/run100k/ckpt_100000.pt")  # eval-ready
```

`load_checkpoint` infers input width and head type from the tensors, so
both v2 (18-feature, scalar value) and v3 (24-feature, WDL) checkpoints
load through the same call.

## Architecture (run100k)

Transformer over 64 square-tokens, d=192, 6 layers, 6 heads, ~2.77M params.
Three heads: from×to policy (4096), per-move Q (4096), value. See
`../prophet/model.py`.
