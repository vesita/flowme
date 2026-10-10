#!/usr/bin/env bash
# L3-m1 损失形状探针 · **第二轮：对比项权重扫描**（PREREG_SHAPE.md §5）
# 用法： systemd-run --user --unit=flowme_m1_shape2 --collect bash scripts/86_runs_shape2.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M1_STEPS:-3000}"
SEED="${M1_SEED:-4321}"

echo "[SHAPE2] start $(date -Is) seed=$SEED steps=$STEPS pid=$$"
for LAM in 0.1 0.3; do
  echo "[SHAPE2] ==== loss=infonce lam=$LAM seed=$SEED start $(date -Is) ===="
  "$PY" stages/86_meme_anchor.py --run "$LAM" "$SEED" --steps "$STEPS" \
        --eval-every 1000 --vision-arch detr --loss infonce \
        > "logs/86_shape_infonce_l${LAM}_s${SEED}.log" 2>&1
  echo "[SHAPE2] loss=infonce lam=$LAM exit=$? $(date -Is)"
done
echo "[SHAPE2-DONE] $(date -Is)"
