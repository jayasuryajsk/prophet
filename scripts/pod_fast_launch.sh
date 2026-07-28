#!/usr/bin/env bash
# Rust-max flagship launch (pod). Fresh 10M net, v3 recipe, fast workers.
#
# PREREQS (once per pod):
#   apt-get update && apt-get install -y build-essential curl
#   curl https://sh.rustup.rs -sSf | sh -s -- -y && . "$HOME/.cargo/env"
#   pip install --break-system-packages torch numpy python-chess requests
#   pip install --break-system-packages ./prophet_core
#
# OPS RULES (learned the hard way — see repo history):
#   * OUT must live on the NETWORK VOLUME (/workspace), never the container
#     disk: a pod restart wipes the container disk and with it the only copy
#     of full_resume.pt (optimizer+buffer). This exact failure killed the
#     v3 run's 60k state.
#   * Pull state OFF the pod too (Mac side: scripts/pull_pod_state.sh) —
#     community pods can lose their GPU at any time.
set -euo pipefail

GAMES="${GAMES:-1000000}"
OUT="${OUT:-/workspace/runs/fast_v3}"
THREADS="${THREADS:-48}"
MEGA="${MEGA:-1024}"
WORKERS="${WORKERS:-1}"   # fast worker PROCESSES (one GIL each; >1 when the
                          # glue lane, not the GPU, is the bottleneck)
RESUME="${RESUME:-}"      # path to full_resume.pt for a warm restart

mkdir -p "$OUT"
cd "$(dirname "$0")/.."

nohup python3 scripts/train_loop.py \
  --games "$GAMES" \
  --fast --fast-threads "$THREADS" --mega-batch "$MEGA" --search-batch 32 \
  --workers "$WORKERS" --worker-device cuda --device cuda \
  ${RESUME:+--resume-full "$RESUME"} \
  --d-model 320 --n-layers 8 --n-heads 8 \
  --sims 32 --candidates 8 \
  --pcr-prob 0.25 --pcr-cheap-sims 12 \
  --max-plies 160 \
  --buffer 300000 \
  --contempt 0.15 --search-contempt 0.15 --win-discount 0.997 \
  --study --schedule --no-eval \
  --out "$OUT" \
  > "$OUT/train.log" 2>&1 &

echo "launched: pid $! -> $OUT/train.log"
echo "watch:    tail -f $OUT/train.log"
