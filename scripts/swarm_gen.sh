#!/usr/bin/env bash
# Bootstrap ONE generation pod into the swarm: setup + launch gen_node.
# usage: PROPHET_STREAM_KEY=... scripts/swarm_gen.sh GEN_HOST GEN_PORT LEARNER_IP CTRL_PORT SINK_PORT [WORKERS]
set -euo pipefail
H="${1:?gen host}"; P="${2:?gen ssh port}"
LIP="${3:?learner ip}"; CTL="${4:?control(http) public port}"; SNK="${5:?sink public port}"
W="${6:-2}"
KEY="${PROPHET_STREAM_KEY:?set PROPHET_STREAM_KEY}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

ssh-keyscan -p "$P" "$H" >> ~/.ssh/known_hosts 2>/dev/null || true
scp -q -P "$P" "$DIR/scripts/pod_setup.sh" "root@$H:/root/"
ssh -p "$P" "root@$H" "bash /root/pod_setup.sh > /root/setup.log 2>&1 && echo SETUP-OK || (tail -3 /root/setup.log; exit 1)"
ssh -p "$P" "root@$H" "cd /workspace/prophet && PROPHET_STREAM_KEY=$KEY nohup python3 scripts/gen_node.py \
  --control http://$LIP:$CTL --sink $LIP:$SNK \
  --workers $W --threads 32 --mega-batch 1024 --device cuda \
  --local-dir /workspace/genmirror > /root/gen.log 2>&1 & sleep 8; head -3 /root/gen.log; exit 0"
echo "GEN NODE LIVE: $H:$P"
