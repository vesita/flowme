#!/usr/bin/env bash
# L3-m1 **诊断①**（PREREG_SEEDSTAB.md）：图片路"平台/逃出"的来源
#   臂① 只图片路（mode=image, loss=none）× seed{1234,5678,9012} × 8000 步   → 判 H1（图片卡自身）vs H2（与文本路共适应）
#   臂② joint + infonce 0.1 × seed 5678 × 20000 步                          → 判 H3（只是慢）
# 用法： systemd-run --user --unit=flowme_m1_seedstab --collect bash scripts/86_runs_seedstab.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "[SEEDSTAB] start $(date -Is) pid=$$"
echo "[SEEDSTAB] ---- 臂① 只图片路（无文本路）----"
for SEED in 1234 5678 9012; do
  echo "[SEEDSTAB] arm1 seed=$SEED start $(date -Is)"
  "$PY" stages/86_meme_anchor.py --run 0 "$SEED" --steps 8000 --eval-every 1000 \
        --vision-arch detr --loss none --mode image \
        > "logs/86_seedstab_imgonly_s${SEED}.log" 2>&1
  echo "[SEEDSTAB] arm1 seed=$SEED exit=$? $(date -Is)"
done
echo "[SEEDSTAB] ---- 臂② joint infonce λ0.1 · s5678 · 20000 步 ----"
echo "[SEEDSTAB] arm2 start $(date -Is)"
"$PY" stages/86_meme_anchor.py --run 0.1 5678 --steps 20000 --eval-every 1000 \
      --vision-arch detr --loss infonce \
      > "logs/86_seedstab_joint_s5678_20k.log" 2>&1
echo "[SEEDSTAB] arm2 exit=$? $(date -Is)"
echo "[SEEDSTAB-DONE] $(date -Is)"
