#!/usr/bin/env bash
# 探索续（仍是 exploratory，不参与判定）：判"步数/学习率"是不是瓶颈。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
rc=0
run() { local n="$1"; shift; echo "[expl2] $(date +%T) start $n"; \
  "$@" > "logs/additivity_${n}.log" 2>&1; local r=$?; \
  echo "[expl2] $(date +%T) done $n rc=$r"; grep -E "AB_METRICS" "logs/additivity_${n}.log" | head -1; \
  [ "$r" -ne 0 ] && rc=1; }
T=experiments/additivity/train_bypass_card.py
CARDS=experiments/additivity/cards
run "e_r32_lr3k_s42" uv run python -u "$T" --mode bypass --seed 42 --rank 32 --lr 3e-3 \
    --out "${CARDS}/negation_byp_r32_lr3k_s42.pt"
run "e_r32_e64_s42" uv run python -u "$T" --mode bypass --seed 42 --rank 32 --epochs 64 \
    --eval-every 16 --out "${CARDS}/negation_byp_r32_e64_s42.pt"
echo "EXPL2_DONE rc=$rc $(date +%T)"
