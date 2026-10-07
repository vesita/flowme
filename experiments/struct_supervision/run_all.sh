#!/usr/bin/env bash
# struct_supervision 全量跑（PREREG §1：3 臂 × 2 seed + 2 个随机标签门禁）
# 用法：systemd-run --user --unit=dtseek-struct-sup --collect \
#   --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
#   --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
#   /usr/bin/bash experiments/struct_supervision/run_all.sh
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EXP=experiments/struct_supervision
LOG=$EXP/logs
mkdir -p "$LOG"

echo "=== selfcheck $(date -Is) ==="
uv run python "$EXP/model.py" --selfcheck > "$LOG/selfcheck.json"
echo "selfcheck ok"

for s in 42 43; do
  for a in A B C; do
    echo "=== $a s$s $(date -Is) ==="
    uv run python "$EXP/train.py" --arm "$a" --seed "$s" 2>&1 | tee "$LOG/${a}_s${s}.log"
  done
done

for s in 42 43; do
  echo "=== A-rand s$s $(date -Is) ==="
  uv run python "$EXP/train.py" --arm A --seed "$s" --randlabel 2>&1 | tee "$LOG/A_s${s}_rand.log"
  echo "=== B-rand s$s $(date -Is) ==="
  uv run python "$EXP/train.py" --arm B --seed "$s" --randlabel 2>&1 | tee "$LOG/B_s${s}_rand.log"
done

echo "=== analyze $(date -Is) ==="
uv run python "$EXP/analyze.py" 2>&1 | tee "$LOG/analyze.log"
echo "=== ALL DONE $(date -Is) ==="
