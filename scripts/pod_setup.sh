#!/usr/bin/env bash
# One-shot pod bring-up: deps + repo + rust build + sanity. Idempotent.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential curl git > /dev/null

if ! command -v cargo >/dev/null; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y -q
fi
. "$HOME/.cargo/env"

pip install -q --no-input --break-system-packages python-chess requests 2>&1 | tail -1 || true

cd /workspace
if [ ! -d prophet ]; then
  git clone -q https://github.com/jayasuryajsk/prophet.git
fi
cd prophet
git fetch -q origin && git checkout -q moonshot-deepreflect && git pull -q

pip install -q --force-reinstall --no-deps --break-system-packages ./prophet_core 2>&1 | tail -1

# sanity: GPU + core + model load + one rust search with history
python3 - <<'EOF'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import numpy as np, chess
import prophet_core
from prophet.model import load_checkpoint
from prophet.searchC import RustBatchedSearcher

m = load_checkpoint("checkpoints/ckpt_060000.pt"); m.eval()
b = chess.Board()
b.push_san("e4"); b.push_san("e5")
s = RustBatchedSearcher(m, budget=256, batch=32, seed=1, contempt=0.1)
mv, spent = s.search(b)
print("search OK:", mv.uci(), "spent", spent)
EOF

nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
free -g | head -2
echo "POD READY"
