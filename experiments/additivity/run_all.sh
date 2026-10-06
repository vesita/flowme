#!/usr/bin/env bash
# E-B 能力可加性：7 个训练跑 + 2 个老卡复核跑，逐个独立进程（PREREG.md 的运行清单）。
# 由 systemd-run --user --unit=dtseek-additivity-byp 启动。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

rc_total=0
run() {
  local name="$1"; shift
  local log="logs/additivity_${name}.log"
  echo "[run] $(date +%T) start ${name} -> ${log}"
  echo "[run] cmd: $*"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[run] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_[123]|AB_METRICS" "$log" | head -6
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

CARDS=experiments/additivity/cards
APR=experiments/additivity/base_aprime.pt

# ---- 主档：base_encoder（PREREG 运行清单）----
for seed in 42 43; do
  run "b_base_s${seed}" uv run python -u experiments/additivity/train_bypass_card.py \
      --mode bypass --base checkpoints/base_encoder.pt --seed "$seed" \
      --out "${CARDS}/negation_bypass_base_s${seed}.pt"
  run "f_base_s${seed}" uv run python -u experiments/additivity/train_bypass_card.py \
      --mode frozen --base checkpoints/base_encoder.pt --seed "$seed" \
      --out "${CARDS}/negation_frozen_base_s${seed}.pt"
done

# ---- 校准档：A′ 基座（接文档 frozen 0.6383 的口径）----
run "f_aprime_s42" uv run python -u experiments/additivity/train_bypass_card.py \
    --mode frozen --base "$APR" --seed 42 \
    --out "${CARDS}/negation_frozen_aprime_s42.pt"
run "b_aprime_s42" uv run python -u experiments/additivity/train_bypass_card.py \
    --mode bypass --base "$APR" --seed 42 \
    --out "${CARDS}/negation_bypass_aprime_s42.pt"
run "b_aprime_s43" uv run python -u experiments/additivity/train_bypass_card.py \
    --mode bypass --base "$APR" --seed 43 \
    --out "${CARDS}/negation_bypass_aprime_s43.pt"

# ---- P1 老卡复核（MultiTaskEngine，A/B/C 三条件 + 逐样本预测哈希）----
run "oldcard_s42" uv run python -u experiments/additivity/eval_old_cards.py \
    --bypass-card "${CARDS}/negation_bypass_base_s42.pt" --seed 42 \
    --out experiments/additivity/old_card_check_s42.json
run "oldcard_s43" uv run python -u experiments/additivity/eval_old_cards.py \
    --bypass-card "${CARDS}/negation_bypass_base_s43.pt" --seed 43 \
    --out experiments/additivity/old_card_check_s43.json

echo "ALL_DONE rc=${rc_total} $(date +%T)"
