#!/bin/bash
# 锚定候选打标 —— 阶段 A：只训头主臂（2 seed）+ 随机标签对照（2 seed），逐个训练后立刻评测。
# 只写 experiments/anchored_select/ 与 logs/anchored_stageA.log。
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EX=experiments/anchored_select
mkdir -p "$EX/cards" "$EX/results"

echo "=== stage A start $(date -Is) ==="

for S in 42 43; do
  echo "--- [main s$S] train"
  uv run python "$EX/run_card.py" --out "$EX/cards/anchored_s$S.pt" \
      --seed "$S" --epochs 12 --steps-per-epoch 150 --samples 8000
  echo "--- [main s$S] eval"
  uv run python "$EX/eval_card.py" --ckpt "$EX/cards/anchored_s$S.pt" \
      --out "$EX/results/main_s$S.json" --tag "main_s$S"
done

for S in 42 43; do
  echo "--- [shuf s$S] train（随机标签对照）"
  uv run python "$EX/run_card.py" --shuf --shuf-seed $((S * 1000 + 7)) \
      --out "$EX/cards/shuf_s$S.pt" --seed "$S" --epochs 12 --steps-per-epoch 150 --samples 8000
  echo "--- [shuf s$S] eval"
  uv run python "$EX/eval_card.py" --ckpt "$EX/cards/shuf_s$S.pt" \
      --out "$EX/results/shuf_s$S.json" --tag "shuf_s$S" --splits test
done

echo "=== STAGE_A_DONE $(date -Is) ==="
