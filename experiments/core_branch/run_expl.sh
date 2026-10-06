#!/usr/bin/env bash
# E-D 探索臂（**不参与 PREREG 判定**，只用于机制）：
#   1) 最优档 mlp 加预算（32ep / 64ep）—— 曲线是新平台还是能逼近 0.87；
#   2) frozen mlp s42 用 --eval-every 0 重跑 —— 定位与 E-B F 档 0.6933 的 0.005 差从哪来。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

rc_total=0
run() {
  local name="$1"; shift
  local log="logs/core_branch_x_${name}.log"
  echo "[run] $(date +%T) start ${name} -> ${log}"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[run] $(date +%T) done  ${name} rc=${r}"
  grep -E "eval_gate_on|EPOCH_EVAL" "$log" | tail -6
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

CARDS=experiments/core_branch/cards
TR="uv run python -u experiments/core_branch/train_branch_card.py"

# frozen 诊断：与 E-B F 档同 eval 口径（eval-every=0）
run "f_mlp_s42_noeval" $TR --mode frozen --branch mlp --seed 42 --eval-every 0 \
    --out "${CARDS}/negation_frozen_mlp_s42_noeval.pt"

# mlp 加预算
for seed in 42 43; do
  run "b_mlp_e32_s${seed}" $TR --mode bypass --branch mlp --seed "$seed" --epochs 32 \
      --out "${CARDS}/negation_mlp_e32_s${seed}.pt"
done
run "b_mlp_e64_s42" $TR --mode bypass --branch mlp --seed 42 --epochs 64 \
    --out "${CARDS}/negation_mlp_e64_s42.pt"

echo "EXPL_DONE rc=${rc_total} $(date +%T)"
