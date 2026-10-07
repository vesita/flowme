#!/usr/bin/env bash
# P19 重训：在**线上核**上冻结核只训头（配方见 PREREG_TRAIN.md §2，顺序写死）
# 用法：systemd-run --user --unit=dtseek-fixneg-r1 --collect \
#   --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
#   --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
#   /usr/bin/bash experiments/fix_negation/run_train.sh r1
# 绝不用 setsid nohup；GPU 忙就排队；产物只写 experiments/fix_negation/cards/。
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EXP=experiments/fix_negation
LOG=$EXP/logs
OUTD=$EXP/cards
mkdir -p "$LOG" "$OUTD"

REC=${1:-r1}
case "$REC" in
  r1)  EPOCHS=40; STEPS=150; SAMPLES=8000 ;;
  # R2 最终定义见 PREREG_TRAIN_CLARIFY2.md：与 R1 同数据(8000)同头同 lr，只把步数 6000→12000
  r2)  EPOCHS=80; STEPS=150; SAMPLES=8000 ;;
  *)   echo "未知配方 $REC（r1|r2）"; exit 2 ;;
esac

echo "=== [$REC] 开始 $(date -Is) epochs=$EPOCHS steps/epoch=$STEPS samples=$SAMPLES ==="
for s in 42 43; do
  echo "=== [$REC s$s] $(date -Is) ==="
  uv run python training/train_task_card.py \
    --card negation --base checkpoints/base_encoder.pt \
    --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --samples "$SAMPLES" \
    --lr-head 1e-3 --batch-size 64 --seed "$s" \
    --out "$OUTD/negation_${REC}_s${s}.pt" \
    2>&1 | tee "$LOG/train_${REC}_s${s}.log"
done
echo "=== [$REC] DONE $(date -Is) ==="
