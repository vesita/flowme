#!/usr/bin/env bash
# P20 三臂 × 2 seed 全量（顺序跑，单卡 8GB；同数据、同步数、同优化器）
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
echo "=== start $(date -Is) pid=$$"
for seed in 42 43; do
  for arm in A B C; do
    echo "=== $arm s$seed $(date -Is)"
    uv run python experiments/core_arch/train.py --arm "$arm" --seed "$seed"
  done
done
echo "=== ALL_DONE $(date -Is)"
