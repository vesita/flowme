#!/usr/bin/env bash
# S87 跑批：A/A 地板 + 主对照（2 臂 × 2 seed）。顺序执行，不并跑（单卡争用会污染测量）。
# 用法：bash stages/87_runs.sh <steps>
set -uo pipefail
cd /home/vesita/coding/my/flowme || exit 1
STEPS="${1:-3000}"
PY="uv run --no-sync python"
FAIL=0

echo "=== [S87] steps=$STEPS 开始 $(date -Is) ==="

for s in 1234 5678; do
  echo "--- [S87] A/A 地板 seed=$s steps=$STEPS ---"
  $PY stages/87_schedule_ab.py --aa const "$s" --steps "$STEPS" 2>&1 | grep -E "^\[AA\]|^\[S87\]|Error|Traceback"
  [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
done

for s in 1234 5678; do
  for sched in const cosine; do
    echo "--- [S87] 主对照 schedule=$sched seed=$s steps=$STEPS ---"
    $PY stages/87_schedule_ab.py --run "$sched" "$s" --steps "$STEPS" 2>&1 | grep -E "^\[RUN-FINAL\]|^\[SPEED\]|Error|Traceback"
    [ "${PIPESTATUS[0]}" -ne 0 ] && FAIL=1
  done
done

# ---- 跑后核验：4 个主产物必须齐 ----
echo "=== [S87] 跑后核验 ==="
for s in 1234 5678; do
  for sched in const cosine; do
    f="logs/87_tok_${sched}_s${s}_st${STEPS}.pt"
    if [ -f "$f" ]; then echo "✅ $f"; else echo "❌ 缺 $f"; FAIL=1; fi
  done
done

if [ "$FAIL" -ne 0 ]; then echo "=== [S87] 批次失败 ==="; exit 1; fi
echo "=== [S87] 批次完成 $(date -Is) ==="
