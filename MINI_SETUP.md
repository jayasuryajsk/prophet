# Running the Prophet Lichess bot (Mac Mini / any machine)

Four commands from a fresh clone. No GPU needed (CPU is fine; M-series is great).

## 1. Clone + deps
```bash
git clone https://github.com/jayasuryajsk/prophet.git && cd prophet
git checkout moonshot-deepreflect
pip3 install torch numpy python-chess requests
```

## 2. Checkpoint
The current serving checkpoint ships in the repo: `checkpoints/ckpt_060000.pt`.
(Newer ones can be scp'd from the training pod into `checkpoints/` later —
the bot takes the path as its first argument.)

## 3. Token
Use the bot account's API token (scopes: bot:play, challenge:read/write).
NEVER commit it.
```bash
export LICHESS_TOKEN=lip_...   # the jayasuryajsk bot token
```

## 4. Run
```bash
python3 scripts/lichess_bot.py checkpoints/ckpt_060000.pt          # the player
python3 scripts/matchmaker.py &                                    # auto-challenges bots when idle
```

That's it. The bot accepts standard challenges (1min+ clocks) from humans and
bots, one game at a time, and reports moves in its stdout log.

## IMPORTANT: one bot process at a time
Only ONE machine may run the bot per token — two event-stream consumers will
fight over games. Before starting here, make sure the pod's copy is stopped:
```bash
ssh root@<pod-ip> -p <port> "pkill -f lichess_bot.py; pkill -f matchmaker.py"
```

## Notes
- Time budgeting is automatic from the clock (brisk: ~1/50th of remaining
  time per move, panic mode under 15s).
- Search: the proven v3 search (`prophet/search.py`).
- To hot-swap a newer checkpoint: stop the bot, point it at the new .pt, start.
