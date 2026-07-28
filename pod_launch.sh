#!/bin/bash
# From-scratch launch of the dueling recipe (v3): dueling Q=V+A head,
# attention-pool WDL readout, moves-left head, TD(8) value targets,
# search-contempt 0.15, truncation-as-draw, conversion-triggered study
# with terminal branches, study telemetry.
cd /root/prophet
# real allowance = cgroup quota, NOT nproc (community pods cap the container
# far below the host's core count)
NCPU=$(awk '{if ($1=="max") print 64; else print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null)
[ -z "$NCPU" ] && NCPU=$(nproc)
CPUW=$(( (NCPU - 5) / 2 )); [ "$CPUW" -lt 4 ] && CPUW=4
echo "cpu quota=$NCPU -> $CPUW cpu workers"
mkdir -p runs
nohup python3 -u scripts/train_loop.py \
  --games 100000 --device cuda \
  --worker-layout cuda:2x64x2,cpu:${CPUW}x24x2 \
  --sims 32 --candidates 8 --buffer 300000 \
  --d-model 320 --n-layers 8 --n-heads 8 --lr 3e-4 \
  --study --schedule --gate 2000 --resign-gate 2000 \
  --warmup 5000 --eval-every 2000 --log-every 100 --no-eval \
  --search-contempt 0.15 \
  --pcr-prob 0.25 --pcr-cheap-sims 12 \
  --out /root/prophet/runs/dueling_v1 \
  > /root/prophet/runs/dueling_v1.log 2>&1 &
echo "launched pid $! — watch: tail -f /root/prophet/runs/dueling_v1.log"
echo "telemetry:      tail -f /root/prophet/runs/dueling_v1/study_telemetry.log"
