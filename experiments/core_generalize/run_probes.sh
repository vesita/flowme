#!/usr/bin/env bash
# core_generalize 阶段 C：探针头训练（核冻结，唯一变量=核）。
# 需要 holdouts.txt（可学性预检后写：每行一个有效 hold-out）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
E=experiments/core_generalize
mkdir -p "$E/logs" "$E/probes"
rc_total=0

if [ ! -s "$E/holdouts.txt" ]; then
  echo "[cg] 缺 $E/holdouts.txt（预检结论未落盘）⇒ 不跑探针"
  exit 2
fi
mapfile -t HOLDOUTS < "$E/holdouts.txt"
echo "[cg] 有效 hold-out: ${HOLDOUTS[*]}"

run() {
  local name="$1"; shift
  local log="$E/logs/${name}.log"
  if [ -f "$E/probes/${name}.json" ]; then
    echo "[cg] skip ${name}（已有产物）"; return 0
  fi
  echo "[cg] $(date +%T) start ${name} -> ${log}"
  /usr/bin/uv run python -u "$@" > "$log" 2>&1
  local r=$?
  echo "[cg] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_budget|SELFTEST_core|SELFTEST_step0|CG_PROBE_METRICS|CG_PROBE_DONE|Traceback|SystemExit" "$log" | tail -6
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

probe() {   # probe <label> <core路径> <seed> <holdout>
  local label="$1" core="$2" seed="$3" h="$4"
  run "${label}_${h}_s${seed}" "$E/train_probe_head.py" \
      --core "$core" --label "$label" --holdout "$h" --seed "$seed" \
      --out "$E/probes/${label}_${h}_s${seed}.json"
}

for h in "${HOLDOUTS[@]}"; do
  # B0 = 未参与任何训练的 base 核（δ 来源 + 起点锚），2 头 seed
  for seed in 42 43; do
    probe "B0" "checkpoints/base_encoder.pt" "$seed" "$h"
  done
  # C1/C3/C5：核 seed 与头 seed 同号配对
  for seed in 42 43; do
    for arm in C1 C3 C5; do
      probe "${arm}_s${seed}" "$E/cores/${arm}_s${seed}.pt" "$seed" "$h"
    done
  done
  # C1x5：仅 seed 42
  probe "C1x5_s42" "$E/cores/C1x5_s42.pt" 42 "$h"
done

echo "CG_PROBES_ALL_DONE rc=$rc_total $(date +%T)"
exit $rc_total
