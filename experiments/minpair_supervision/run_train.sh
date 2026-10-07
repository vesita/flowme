#!/bin/bash
# P21 训练驱动：参数为 "arm:seed:rand" 三元组（rand=0/1），顺序执行；日志同时落 logs/。
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
LOG="logs/p21-train-$(date +%H%M%S)-$$.log"
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1
echo "=== [stream start] $* → $LOG"
for spec in "$@"; do
  IFS=: read -r arm seed rand <<< "$spec"
  extra=""
  if [ "$rand" = "1" ]; then extra="--randlabel"; fi
  echo "=== [run] arm=$arm seed=$seed rand=$rand  $(date -Is)"
  uv run python experiments/minpair_supervision/train.py \
    --arm "$arm" --seed "$seed" --steps 1800 $extra
done
echo "=== [stream done] $*  $(date -Is)"
