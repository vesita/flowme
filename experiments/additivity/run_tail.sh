#!/usr/bin/env bash
# 收尾：补跑 s43 老卡复核（前一次因 bypass.py 的子模块递归 bug 失败，已修）+ 探索臂。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
echo "[tail] $(date +%T) start oldcard_s43(rerun)"
uv run python -u experiments/additivity/eval_old_cards.py \
  --bypass-card experiments/additivity/cards/negation_bypass_base_s43.pt --seed 43 \
  --out experiments/additivity/old_card_check_s43.json > logs/additivity_oldcard_s43.log 2>&1
echo "[tail] $(date +%T) oldcard_s43 rc=$?"
bash experiments/additivity/run_exploratory.sh
echo "TAIL_DONE $(date +%T)"
