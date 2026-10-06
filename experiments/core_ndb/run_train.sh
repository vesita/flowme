#!/usr/bin/env bash
# core_ndb 阶段 2：4 次温启动联合训练（臂 C 核级记忆 × 2 seed + 臂 L 卡级 NDB × 2 seed）
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
for arm in C L; do
  for seed in 42 43; do
    echo "=== arm=$arm seed=$seed start $(date -Is) ==="
    uv run python -u experiments/core_ndb/train_core_ndb.py \
      --arm "$arm" --seed "$seed" \
      --out "experiments/core_ndb/cards/p2_${arm}_s${seed}.pt" \
      --metrics "experiments/core_ndb/results/p2_${arm}_s${seed}.json"
    echo "=== arm=$arm seed=$seed done $(date -Is) ==="
  done
done
echo "ALL_TRAIN_DONE"
