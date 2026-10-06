#!/usr/bin/env bash
# core_generalize 阶段 B：四个核臂（C1/C3/C5 × seed{42,43} + C1x5 仅 seed42）
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
  grep -E "SELFTEST|CG_CORE_METRICS|CG_CORE_DONE|Traceback|SystemExit|作废" "$log" | tail -8
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

for seed in 42 43; do
  for arm in C1 C3 C5; do
    run "${arm}_s${seed}" "$E/train_core_arm.py" --arm "$arm" --seed "$seed" \
        --out "$E/cores/${arm}_s${seed}.pt" --metrics "$E/cores/${arm}_s${seed}_metrics.json"
  done
done
# 数据量控制臂：只 sentiment、步数 ×5（80 ep × 84 = 6720），仅 seed 42
run "C1x5_s42" "$E/train_core_arm.py" --arm C1x5 --seed 42 \
    --out "$E/cores/C1x5_s42.pt" --metrics "$E/cores/C1x5_s42_metrics.json"

echo "CG_CORES_ALL_DONE rc=$rc_total $(date +%T)"
exit $rc_total
