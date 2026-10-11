#!/usr/bin/env bash
# L3-m2 **全量跑批**（PREREG_ZH_EN.md）：3 组 × 2 seed × 8000 步
#   ① none  | AaBb          零假设（两路无联系 ⇒ 期望 Δ≈0）
#   ② none  | AaBb+AbBa     用户的映射方案（只靠翻译映射，无相似度损失）
#   ③ infonce(0.1) | AaBb+AbBa   映射 + 显式对比项（看显式项是否还有增量）
#   用法： systemd-run --user --unit=flowme_m2_full --collect bash scripts/86g_runs_full.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M2_STEPS:-8000}"

echo "[M2FULL] start $(date -Is) steps=$STEPS pid=$$"
run() {  # $1=maps  $2=loss  $3=lam  $4=seed
  echo "[M2FULL] ==== maps=$1 loss=$2 lam=$3 seed=$4 start $(date -Is) ===="
  "$PY" stages/86g_zh_en.py --run "$3" "$4" --steps "$STEPS" --eval-every 1000 \
        --maps "$1" --loss "$2" \
        > "logs/86g_full_${2}_${1//,/_}_l${3}_s${4}.log" 2>&1
  echo "[M2FULL] maps=$1 loss=$2 lam=$3 seed=$4 exit=$? $(date -Is)"
}

for S in 1234 5678; do run AaBb none 0 "$S"; done
for S in 1234 5678; do run AaBb,AbBa none 0 "$S"; done
for S in 1234 5678; do run AaBb,AbBa infonce 0.1 "$S"; done
echo "[M2FULL-DONE] $(date -Is)"
