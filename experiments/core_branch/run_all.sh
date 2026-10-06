#!/usr/bin/env bash
# E-D 并联分支：6 个 frozen 空测试②跑 + 6 个主档训练跑 + 6 个老卡 A/B/C 复核跑。
# 由 systemd-run --user --unit=dtseek-core-branch 启动（PREREG.md §6 运行清单）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

rc_total=0
run() {
  local name="$1"; shift
  local log="logs/core_branch_${name}.log"
  echo "[run] $(date +%T) start ${name} -> ${log}"
  echo "[run] cmd: $*"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[run] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_[123]|AB_METRICS" "$log" | head -8
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

CARDS=experiments/core_branch/cards
TR="uv run python -u experiments/core_branch/train_branch_card.py"
EV="uv run python -u experiments/core_branch/eval_old_cards_branch.py"

# ---- 空测试②：零分支 frozen 档（分支在场、冻结在零、开闸）----
for br in mlp block1 block2; do
  for seed in 42 43; do
    run "f_${br}_s${seed}" $TR --mode frozen --branch "$br" \
        --base checkpoints/base_encoder.pt --seed "$seed" \
        --out "${CARDS}/negation_frozen_${br}_s${seed}.pt"
  done
done

# ---- 主档：3 档并联分支 × 2 seed（PREREG 预算 1344 步）----
for br in mlp block1 block2; do
  for seed in 42 43; do
    run "b_${br}_s${seed}" $TR --mode bypass --branch "$br" \
        --base checkpoints/base_encoder.pt --seed "$seed" \
        --out "${CARDS}/negation_${br}_s${seed}.pt"
  done
done

# ---- P1 老卡复核（MultiTaskEngine，A/B/C 三条件 + 逐样本预测哈希）----
for br in mlp block1 block2; do
  for seed in 42 43; do
    run "oldcard_${br}_s${seed}" $EV \
        --branch-card "${CARDS}/negation_${br}_s${seed}.pt" --seed "$seed" \
        --out "experiments/core_branch/old_card_check_${br}_s${seed}.json"
  done
done

echo "ALL_DONE rc=${rc_total} $(date +%T)"
