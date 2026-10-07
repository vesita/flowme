#!/usr/bin/env bash
# P7 收敛档（PREREG rev.B 敏感性，单列，不替换主表）：steps=8000。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
rc=0
run() {
  local name="$1"; shift
  local log="logs/sm7_ext_${name}.log"
  echo "[run] $(date +%T) start ${name}"; "$@" > "$log" 2>&1; local r=$?
  echo "[run] $(date +%T) done  ${name} rc=${r}"; grep -E "^\[train\]|^\[eval\]|^\[done\]" "$log" | tail -8
  [ "$r" -ne 0 ] && rc=1
  return 0
}
TR="uv run python -u experiments/sentence_mode/train.py"
run MASK_s42_8k  $TR --arm MASK --seed 42 --steps 8000
run MASK_s43_8k  $TR --arm MASK --seed 43 --steps 8000
run KEEP_s42_8k  $TR --arm KEEP --seed 42 --steps 8000
echo "EXT_DONE rc=${rc} $(date +%T)"
