#!/usr/bin/env bash
# L3-m2 第二轮：**AbBa 权重扫描 α**（PREREG_ZH_EN.md §7）· loss=none · 8000 步 · 2 seed/档
#   α=1.0 那一档**复用第一轮的 run**，不在此重跑。
#   用法： systemd-run --user --unit=flowme_m2_alpha --collect bash scripts/86g_runs_alpha.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M2_STEPS:-8000}"

echo "[ALPHA] start $(date -Is) steps=$STEPS pid=$$"
for W in 0.25 0.5; do
  for S in 1234 5678; do
    echo "[ALPHA] ==== α=$W seed=$S start $(date -Is) ===="
    "$PY" stages/86g_zh_en.py --run 0 "$S" --steps "$STEPS" --eval-every 1000 \
          --maps AaBb,AbBa --loss none --abba-weight "$W" \
          > "logs/86g_alpha_a${W}_s${S}.log" 2>&1
    echo "[ALPHA] α=$W seed=$S exit=$? $(date -Is)"
  done
done
echo "[ALPHA-DONE] $(date -Is)"
