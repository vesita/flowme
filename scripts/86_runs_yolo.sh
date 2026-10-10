#!/usr/bin/env bash
# L3-m1 **附加臂**：YOLO 式视觉槽头（`--vision-arch yolo`），λ=0.5 × 2 seed。
# ★ 不参与主判词（PREREG 订正 2：主档 = detr）；结果文件带 arch 前缀，不会覆盖主档。
# 用法： systemd-run --user --unit=flowme_m1_yolo --collect bash scripts/86_runs_yolo.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M1_STEPS:-10000}"

echo "[RUNNER-YOLO] start $(date -Is) steps=$STEPS"
for cfg in 0.5:1234 0.5:5678; do
  LAM="${cfg%%:*}"
  SEED="${cfg##*:}"
  echo "[RUNNER-YOLO] ==== lam=$LAM seed=$SEED start $(date -Is) ===="
  "$PY" stages/86_meme_anchor.py --run "$LAM" "$SEED" --steps "$STEPS" \
        --eval-every 1000 --vision-arch yolo \
        > "logs/86_run_yolo_l${LAM}_s${SEED}.log" 2>&1
  echo "[RUNNER-YOLO] lam=$LAM seed=$SEED exit=$? $(date -Is)"
done
echo "[YOLO-DONE] $(date -Is)"
