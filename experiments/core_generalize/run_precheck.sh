#!/usr/bin/env bash
# core_generalize 阶段 A：hold-out 可学性预检（PRE-idiom / PRE-ownership × 2 seed）
# 只写 experiments/core_generalize/ 与 logs/（本目录下）
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
E=experiments/core_generalize
mkdir -p "$E/logs" "$E/cores"
rc_total=0

run() {
  local name="$1"; shift
  local log="$E/logs/${name}.log"
  if [ -f "$E/cores/${name}_metrics.json" ]; then
    echo "[cg] skip ${name}（已有产物）"; return 0
  fi
  echo "[cg] $(date +%T) start ${name} -> ${log}"
  /usr/bin/uv run python -u "$@" > "$log" 2>&1
  local r=$?
  echo "[cg] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST|CG_CORE_METRICS|CG_CORE_DONE|Traceback|SystemExit|作废" "$log" | tail -6
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

for seed in 42 43; do
  for h in idiom ownership; do
    run "PRE_${h}_s${seed}" "$E/train_core_arm.py" --arm PRE --holdout "$h" --seed "$seed" \
        --skip-init-check \
        --out "$E/cores/PRE_${h}_s${seed}.pt" \
        --metrics "$E/cores/PRE_${h}_s${seed}_metrics.json"
  done
done

echo "CG_PRECHECK_ALL_DONE rc=$rc_total $(date +%T)"
exit $rc_total
