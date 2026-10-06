#!/usr/bin/env bash
# core_keep：保真档 F（先跑，验脚本）→ J0~J3 × seed{42,43} → 统一评测
# 只写 experiments/core_keep/ 与 logs/corekeep_*；不碰 src/ training/ tests/ dev-notes/
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
CARDS=experiments/core_keep/cards
mkdir -p "$CARDS"
rc_total=0

run() {
  local name="$1"; shift
  local log="logs/corekeep_${name}.log"
  echo "[corekeep] $(date +%T) start ${name} -> ${log}"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[corekeep] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_[0-9]|CK_METRICS|Traceback|SystemExit|Error" "$log" | tail -8
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

# 0) 保真对照 F：随机初始化 5 卡联合，应复现 checkpoints/arm_neg5_seed42 的口径
run "F_s42" uv run python -u experiments/core_keep/train_core_keep.py \
    --arm F --warm off --seed 42 \
    --out "$CARDS/F_s42.pt" --metrics "$CARDS/F_s42_metrics.json"

# 1) 温启动四档 × 2 seed（PREREG §3 主判定）
for seed in 42 43; do
  for arm in J0 J1 J2 J3; do
    run "${arm}_s${seed}" uv run python -u experiments/core_keep/train_core_keep.py \
        --arm "$arm" --seed "$seed" \
        --out "$CARDS/${arm}_s${seed}.pt" --metrics "$CARDS/${arm}_s${seed}_metrics.json"
  done
done

# 2) 追加档 J1c：从头联合 + 蒸馏（对照 = 已有 checkpoints/arm_neg5_seed{42,43}，不重训）
for seed in 42 43; do
  run "J1c_s${seed}" uv run python -u experiments/core_keep/train_core_keep.py \
      --arm J1 --warm off --seed "$seed" \
      --out "$CARDS/J1c_s${seed}.pt" --metrics "$CARDS/J1c_s${seed}_metrics.json"
done

# 3) 统一评测（基线现算 + P1/P2 判定）
run "eval" uv run python -u experiments/core_keep/eval_core_keep.py \
    --arms J0,J1,J2,J3,J1c,F

echo "COREKEEP_ALL_DONE rc=$rc_total $(date +%T)"
exit $rc_total
