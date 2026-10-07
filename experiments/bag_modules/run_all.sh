#!/usr/bin/env bash
# bag_modules 全量跑（PREREG §5：4 臂 × 2 seed + 2 随机标签门禁 + 3 杀模块消融 ×2 seed）
# 用法：systemd-run --user --unit=dtseek-bagmod --collect \
#   --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
#   --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
#   /usr/bin/bash experiments/bag_modules/run_all.sh
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EXP=experiments/bag_modules
LOG=$EXP/logs
mkdir -p "$LOG"

echo "=== selfcheck $(date -Is) ==="
uv run python "$EXP/model.py" --selfcheck > "$LOG/selfcheck.json"
echo "selfcheck ok"

# 主表：四臂 × 2 seed
for s in 42 43; do
  for a in A U P UP; do
    echo "=== $a s$s $(date -Is) ==="
    uv run python "$EXP/train.py" --arm "$a" --seed "$s" 2>&1 | tee "$LOG/${a}_s${s}.log"
  done
done

# B3 门禁：U 随机标签 × 2 seed
for s in 42 43; do
  echo "=== U-rand s$s $(date -Is) ==="
  uv run python "$EXP/train.py" --arm U --seed "$s" --randlabel 2>&1 | tee "$LOG/U_s${s}_rand.log"
done

# 机制：杀模块消融（只留一个模块）× 2 seed
for s in 42 43; do
  for m in type role cls; do
    echo "=== U-only-$m s$s $(date -Is) ==="
    uv run python "$EXP/train.py" --arm U --seed "$s" --mods "$m" 2>&1 | tee "$LOG/U_s${s}_m${m}.log"
  done
done

echo "=== ALL DONE $(date -Is) ==="
