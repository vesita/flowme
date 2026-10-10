#!/usr/bin/env bash
# L3-m1 **补救轮**（PREREG §7 / 订正 11）：6 run 在 10000 步上限时明显未饱和 ⇒ 加步数。
# 新 seed（9012）⇒ 与主档 6 run 的文件名不冲突；结果不参与主档判词，在 REPORT 里单列。
# 用法： systemd-run --user --unit=flowme_m1_remedy --collect bash scripts/86_runs_remedy.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M1_STEPS:-30000}"
SEED="${M1_SEED:-9012}"
LAM="${M1_LAM:-0.5}"

echo "[REMEDY] start $(date -Is) lam=$LAM seed=$SEED steps=$STEPS"
"$PY" stages/86_meme_anchor.py --run "$LAM" "$SEED" --steps "$STEPS" \
      --eval-every 2000 --vision-arch detr \
      > "logs/86_run_remedy_l${LAM}_s${SEED}.log" 2>&1
echo "[REMEDY] exit=$? $(date -Is)"
echo "[REMEDY-DONE] $(date -Is)"
