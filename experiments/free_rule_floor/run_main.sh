#!/usr/bin/env bash
# free_rule_floor 主跑：selfcheck + F0 复现臂 A/U/UP × 2 seed（PREREG §5）
# 用法：systemd-run --user --unit=dtseek-frf1 --collect \
#   --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
#   --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
#   /usr/bin/bash experiments/free_rule_floor/run_main.sh
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EXP=experiments/free_rule_floor
LOG=$EXP/logs
mkdir -p "$LOG"

echo "=== selfcheck $(date -Is) ==="
uv run python "$EXP/narm.py" --selfcheck > "$LOG/selfcheck.json" 2>"$LOG/selfcheck.err"
grep -q '"ok": true' "$LOG/selfcheck.json"
echo "selfcheck ok"

for s in 42 43; do
  for a in A U UP; do
    echo "=== $a s$s $(date -Is) ==="
    uv run python "$EXP/train.py" --arm "$a" --seed "$s" 2>&1 | tee "$LOG/${a}_s${s}.log"
  done
done
echo "=== MAIN DONE $(date -Is) ==="
