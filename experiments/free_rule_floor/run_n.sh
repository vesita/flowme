#!/usr/bin/env bash
# free_rule_floor 对照跑：新臂 N × 2 seed + 负对照 U-rand × 2 seed（PREREG §1/§2）
# 用法：systemd-run --user --unit=dtseek-frf2 --collect \
#   --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
#   --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
#   /usr/bin/bash experiments/free_rule_floor/run_n.sh
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EXP=experiments/free_rule_floor
LOG=$EXP/logs
mkdir -p "$LOG"

for s in 42 43; do
  echo "=== N s$s $(date -Is) ==="
  uv run python "$EXP/train.py" --arm N --seed "$s" 2>&1 | tee "$LOG/N_s${s}.log"
done
for s in 42 43; do
  echo "=== U-rand s$s $(date -Is) ==="
  uv run python "$EXP/train.py" --arm U --seed "$s" --randlabel 2>&1 | tee "$LOG/U_s${s}_rand.log"
done
echo "=== N+RAND DONE $(date -Is) ==="
