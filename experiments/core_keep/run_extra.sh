#!/usr/bin/env bash
# core_keep 追加：λ_kd 升档阶梯（PREREG §4.1，s42 探索性）+ 复评
# 必须在 run_all.sh 结束之后启动（避免与主跑抢 GPU / 抢同一份脚本）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
CARDS=experiments/core_keep/cards
rc_total=0

run() {
  local name="$1"; shift
  local log="logs/corekeep_${name}.log"
  echo "[corekeep] $(date +%T) start ${name} -> ${log}"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[corekeep] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_[0-9]|CK_METRICS|Traceback|SystemExit|Error" "$log" | tail -8
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

run "J1l4_s42" uv run python -u experiments/core_keep/train_core_keep.py \
    --arm J1 --seed 42 --kd-weight 4 \
    --out "$CARDS/J1l4_s42.pt" --metrics "$CARDS/J1l4_s42_metrics.json"

run "J1l16_s42" uv run python -u experiments/core_keep/train_core_keep.py \
    --arm J1 --seed 42 --kd-weight 16 \
    --out "$CARDS/J1l16_s42.pt" --metrics "$CARDS/J1l16_s42_metrics.json"

run "eval2" uv run python -u experiments/core_keep/eval_core_keep.py \
    --arms J0,J1,J2,J3,J1c,F,J1l4,J1l16 --out experiments/core_keep/results.json

echo "COREKEEP_EXTRA_DONE rc=$rc_total $(date +%T)"
exit $rc_total
