# Prophet bot on the Mac Mini (M4) — full Phase C setup

Expected result: the bot serving at ~2,000-3,000 forwards/s on the M4 GPU —
blitz games at 6-10k forwards per move, the strongest Prophet ever fielded.

## 0. One-time prerequisites (~5 min)
```bash
xcode-select --install          # compiler toolchain (skip if present)
curl https://sh.rustup.rs -sSf | sh -s -- -y     # rust toolchain
source "$HOME/.cargo/env"
```

## 1. Clone + python env
```bash
git clone https://github.com/jayasuryajsk/prophet.git && cd prophet
git checkout moonshot-deepreflect
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy python-chess requests
```

## 2. Build the Rust search core
```bash
# the env var is only needed on very new Pythons (3.13+); harmless otherwise
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install ./prophet_core
python3 -c "import prophet_core; print(hasattr(prophet_core, 'BatchSearch'))"  # True
```

## 3. IMPORTANT — stop the pod's bot first (one process per token, ever)
Easiest: ask Claude in the work session to stop it. Manual:
```bash
ssh root@67.223.143.80 -p 19686 "pkill -f lichess_bot.py; pkill -f matchmaker.py"
```
(port changes if the pod restarts — check the RunPod console's SSH tab)

## 4. Run
```bash
export LICHESS_TOKEN=lip_...        # the bot token — never commit it
python3 scripts/lichess_bot.py checkpoints/ckpt_060000.pt   # auto-detects MPS
# in a second terminal (same venv):
python3 scripts/matchmaker.py &
```
The startup log prints the measured forwards/s — on the M4 expect 1,500-3,000.
Watch it play: https://lichess.org/@/jayasuryajsk/tv

## 5. Newer checkpoints (the bot gets smarter as the pod trains)
```bash
scp -P 19686 root@67.223.143.80:/root/prophet/runs/dueling_v1/ckpt_0NN000.pt checkpoints/
# restart the bot pointing at the new file
```
Or `git pull` — milestone checkpoints get committed to the repo periodically.

## Troubleshooting
- `externally-managed-environment` pip error → you skipped the venv (step 1)
- bot plays but slowly → check the startup log says "mps" pathway (torch
  built with MPS ships by default on Apple Silicon)
- two bots fighting over games → you forgot step 3
