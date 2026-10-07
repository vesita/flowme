#!/usr/bin/env bash
# P7 训练全量：3 臂 × 2 seed + MASK 随机标签负对照 × 2 seed（PREREG §6）。
# 由 systemd-run --user --unit=sm7-train 启动（PREREG §8 运行清单）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

rc_total=0
run() {
  local name="$1"; shift
  local log="logs/sm7_train_${name}.log"
  echo "[run] $(date +%T) start ${name} -> ${log}"
  echo "[run] cmd: $*"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[run] $(date +%T) done  ${name} rc=${r}"
  grep -E "^\[eval\]|^\[done\]" "$log" | tail -6
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

TR="uv run python -u experiments/sentence_mode/train.py"

for arm in KEEP MASK MIX; do
  for seed in 42 43; do
    run "${arm}_s${seed}" $TR --arm "$arm" --seed "$seed"
  done
done
for seed in 42 43; do
  run "MASK_s${seed}_rand" $TR --arm MASK --seed "$seed" --randlabel
done

echo "ALL_DONE rc=${rc_total} $(date +%T)"
