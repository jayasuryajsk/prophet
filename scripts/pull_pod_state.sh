#!/usr/bin/env bash
# Mac-side state mirror: periodically pull training state OFF the pod so a
# dead pod/GPU costs minutes, not the run. Convenience notifier only — the
# system of record is the pod's network volume (see pod_fast_launch.sh).
#
# usage: scripts/pull_pod_state.sh HOST PORT [REMOTE_OUT] [LOCAL_DIR]
set -euo pipefail

HOST="${1:?host}"
PORT="${2:?port}"
REMOTE="${3:-/workspace/runs/fast_v3}"
LOCAL="${4:-$HOME/prophet/runs/pod_mirror_fast}"

mkdir -p "$LOCAL"
while true; do
  scp -q -P "$PORT" "root@$HOST:$REMOTE/metrics.csv" "$LOCAL/" 2>/dev/null || true
  scp -q -P "$PORT" "root@$HOST:$REMOTE/train.log" "$LOCAL/" 2>/dev/null || true
  scp -q -P "$PORT" "root@$HOST:$REMOTE/study_telemetry.log" "$LOCAL/" 2>/dev/null || true
  scp -q -P "$PORT" "root@$HOST:$REMOTE/broker_stats.log" "$LOCAL/" 2>/dev/null || true
  # newest milestone ckpts (cheap, 39MB each)
  for f in $(ssh -p "$PORT" "root@$HOST" "ls -t $REMOTE/ckpt_*.pt 2>/dev/null | head -2" 2>/dev/null); do
    b="$(basename "$f")"
    [ -f "$LOCAL/$b" ] || scp -q -P "$PORT" "root@$HOST:$f" "$LOCAL/" || true
  done
  # full resume state (optimizer+buffer, ~4GB) — only when its mtime changed
  rts="$(ssh -p "$PORT" "root@$HOST" "stat -c %Y $REMOTE/full_resume.pt 2>/dev/null" 2>/dev/null || echo 0)"
  lts="$(cat "$LOCAL/.full_resume.mtime" 2>/dev/null || echo -1)"
  if [ "$rts" != "0" ] && [ "$rts" != "$lts" ]; then
    echo "$(date '+%H:%M') pulling full_resume.pt ($rts)..."
    scp -q -P "$PORT" "root@$HOST:$REMOTE/full_resume.pt" "$LOCAL/full_resume.pt.tmp" \
      && mv "$LOCAL/full_resume.pt.tmp" "$LOCAL/full_resume.pt" \
      && echo "$rts" > "$LOCAL/.full_resume.mtime"
  fi
  sleep 300
done
