#!/usr/bin/env bash
# 能力筛查 第 2 步：4 老卡 × {frozen, bypass} × {42,43} = 16 跑
# （negation 两档复用 experiments/additivity/cards/ 里同配方同 seed 的 ckpt，不重跑）
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
CARDS=experiments/capability_map/cards
T=experiments/capability_map/train_cards.py
rc_total=0
run() {
  local name="$1"; shift
  local log="logs/capmap_${name}.log"
  echo "[capmap] $(date +%T) start ${name} -> ${log}"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[capmap] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_[123]|AB_METRICS|Error|error" "$log" | head -8
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}
for cap in pronoun sentiment relation person; do
  extra=""
  [ "$cap" = "sentiment" ] && extra="--samples 32000"
  for seed in 42 43; do
    for mode in frozen bypass; do
      run "${cap}_${mode}_s${seed}" uv run python -u "$T" \
          --mode "$mode" --card "$cap" --seed "$seed" $extra \
          --out "${CARDS}/${cap}_${mode}_s${seed}.pt"
    done
  done
done
echo "CAPMAP_TRAIN_DONE rc=${rc_total} $(date +%T)"
exit $rc_total
