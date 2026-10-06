#!/usr/bin/env bash
# E-B 探索臂（PREREG 登记的"失败后探索"，**不参与判定**）：判机制是
# 秩/步数不够，还是插入点/优化本质不足。全部在主基座 base_encoder、seed 42。
# 另含一个机制探针（训练好的旁路改了多少表征、关闸后指标掉不掉）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

rc_total=0
run() {
  local name="$1"; shift
  local log="logs/additivity_${name}.log"
  echo "[expl] $(date +%T) start ${name} -> ${log}"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[expl] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_1 |SELFTEST_3_verdict|AB_METRICS|PROBE" "$log" | head -4
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

CARDS=experiments/additivity/cards
T=experiments/additivity/train_bypass_card.py

run "probe_b_s42" uv run python -u experiments/additivity/probe_bypass.py \
    --card "${CARDS}/negation_bypass_base_s42.pt" --seed 42 \
    --out experiments/additivity/probe_bypass_base_s42.json

# 秩翻倍
run "e_r32_s42" uv run python -u "$T" --mode bypass --seed 42 --rank 32 \
    --out "${CARDS}/negation_byp_r32_s42.pt"
# 步数翻倍（每 8 epoch 打一次 EPOCH_EVAL，看还在不在爬）
run "e_r16_2x_s42" uv run python -u "$T" --mode bypass --seed 42 --rank 16 \
    --epochs 32 --eval-every 8 --out "${CARDS}/negation_byp_r16_e32_s42.pt"
# 秩×步数都翻倍
run "e_r32_2x_s42" uv run python -u "$T" --mode bypass --seed 42 --rank 32 \
    --epochs 32 --eval-every 8 --out "${CARDS}/negation_byp_r32_e32_s42.pt"
# 全秩对照（rank=128 = 不做低秩瓶颈，仍只训这 4 个 delta，基座冻结）
run "e_r128_s42" uv run python -u "$T" --mode bypass --seed 42 --rank 128 \
    --out "${CARDS}/negation_byp_r128_s42.pt"

echo "EXPL_DONE rc=${rc_total} $(date +%T)"
