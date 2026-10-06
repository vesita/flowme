#!/bin/bash
# 阶段 B：P4 温启动联合臂（control 口径，2 seed）+ 联合产物在三集上的附带评测。
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EX=experiments/anchored_select
mkdir -p "$EX/cards" "$EX/results"

echo "=== stage B start $(date -Is) ==="
for S in 42 43; do
  echo "--- [joint s$S] train（4 老卡头冻结 + 核可训 + anchored_sel 头，1344 步）"
  uv run python "$EX/train_joint.py" --seed "$S" \
      --out "$EX/cards/joint_s$S.pt" --metrics "$EX/results/joint_train_s$S.json"
done
for S in 42 43; do
  echo "--- [joint s$S] eval（三集附带，不参与判据）"
  uv run python "$EX/eval_card.py" --joint --task anchored_sel \
      --ckpt "$EX/cards/joint_s$S.pt" --out "$EX/results/joint_eval_s$S.json" --tag "joint_s$S"
done
echo "=== STAGE_B_DONE $(date -Is) ==="
