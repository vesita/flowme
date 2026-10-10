#!/usr/bin/env bash
# L3-m1 **损失形状探针**（PREREG_SHAPE.md）：4 配置 × seed 4321 × 3000 步，串行。
# seed 4321 不在 PRIMARY_SEEDS=(1234,5678) ⇒ 判词会自动把这些列成"非主档读数"，不污染主档。
# 用法： systemd-run --user --unit=flowme_m1_shape --collect bash scripts/86_runs_shape.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M1_STEPS:-3000}"
SEED="${M1_SEED:-4321}"

echo "[SHAPE] start $(date -Is) seed=$SEED steps=$STEPS pid=$$"
# cfg = 损失形状:lam:额外参数
for cfg in \
  "none:0" \
  "align:0.5" \
  "infonce:1.0" \
  "infonce_var:1.0" ; do
  LOSS="${cfg%%:*}"
  LAM="${cfg##*:}"
  echo "[SHAPE] ==== loss=$LOSS lam=$LAM seed=$SEED start $(date -Is) ===="
  "$PY" stages/86_meme_anchor.py --run "$LAM" "$SEED" --steps "$STEPS" \
        --eval-every 1000 --vision-arch detr --loss "$LOSS" \
        > "logs/86_shape_${LOSS}_l${LAM}_s${SEED}.log" 2>&1
  echo "[SHAPE] loss=$LOSS lam=$LAM exit=$? $(date -Is)"
done
echo "[SHAPE-DONE] $(date -Is)"
