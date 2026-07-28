#!/bin/bash
# RunPod one-time setup for the dueling-recipe run.
#
# 1) from the Mac (fill in PORT/IP):
#    rsync -az --exclude runs --exclude .git --exclude target \
#      -e "ssh -p PORT -i ~/.ssh/id_ed25519" \
#      /Users/macstudio/prophet-moonshot/ root@IP:/root/prophet/
# 2) on the pod:  bash /root/prophet/pod_setup.sh
set -e
cd /root/prophet
apt-get update -qq && apt-get install -y -qq build-essential curl stockfish >/dev/null
command -v cargo >/dev/null 2>&1 || {
  curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null
}
. "$HOME/.cargo/env" 2>/dev/null || true
pip -q install --break-system-packages python-chess numpy maturin
# rust board core (~11x board ops); fastboard falls back to python-chess if absent
pip -q install --break-system-packages ./prophet_core || (cd prophet_core && maturin develop --release) || \
  echo "RUST BUILD FAILED — continuing on python-chess fallback (slower)"
python3 - <<'EOF'
import torch
try:
    import prophet_core
    print("rust core:", hasattr(prophet_core, "Board"))
except ImportError:
    print("rust core: MISSING (python-chess fallback, slower)")
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import chess.engine
e = chess.engine.SimpleEngine.popen_uci("stockfish"); e.quit(); print("stockfish: OK")
EOF
echo "vCPUs: $(nproc)"
echo "setup done — launch: bash /root/prophet/pod_launch.sh"
