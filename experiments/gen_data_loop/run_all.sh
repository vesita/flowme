#!/usr/bin/env bash
# Phase 2：D4 两臂对照训练 + 评测（PREREG_PHASE2.md §8 运行清单）。
# 由 systemd-run --user --unit=dtseek-genphase2 启动（会话重启不掉）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

PY=/home/vesita/coding/my/DTSeek/.venv/bin/python
TR="experiments/gen_data_loop/run_card.py"
EV="experiments/gen_data_loop/eval_card.py"
CARDS=experiments/gen_data_loop/cards
RES=experiments/gen_data_loop/results
SPLITS=natural,adversarial,ctrl_adversarial,test,ctrl_test

rc_total=0
run() {
  local name="$1"; shift
  local log="logs/genphase2_${name}.log"
  echo "[run] $(date +%T) start ${name} -> ${log}"
  echo "[run] cmd: $*"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[run] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_1|SELFTEST_DATA|AB_METRICS" "$log" | head -6
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

# ---- 1) 主档：P / C × 2 seed（同配方、同步数、同 seed，唯一变量 = 数据来源）----
for arm in P C; do
  for seed in 42 43; do
    run "train_${arm}_s${seed}" "$PY" -u "$TR" \
        --arm "$arm" --seed "$seed" --out "${CARDS}/cand_${arm}_s${seed}.pt" \
        --epochs 12 --steps-per-epoch 150 --batch-size 64 --lr-head 1e-3 --samples 8000
    run "eval_${arm}_s${seed}" "$PY" -u "$EV" \
        --ckpt "${CARDS}/cand_${arm}_s${seed}.pt" --tag "${arm}_s${seed}" \
        --splits "$SPLITS" --out "${RES}/eval_${arm}_s${seed}.json"
  done
done

# ---- 2) 随机标签对照（V-bonus，shuf-seed = seed×1000+7）----
for arm in P C; do
  for seed in 42 43; do
    run "train_${arm}_shuf_s${seed}" "$PY" -u "$TR" \
        --arm "$arm" --seed "$seed" --shuf --shuf-seed "$((seed*1000+7))" \
        --out "${CARDS}/cand_${arm}_shuf_s${seed}.pt" \
        --epochs 12 --steps-per-epoch 150 --batch-size 64 --lr-head 1e-3 --samples 8000
    run "eval_${arm}_shuf_s${seed}" "$PY" -u "$EV" \
        --ckpt "${CARDS}/cand_${arm}_shuf_s${seed}.pt" --tag "${arm}_shuf_s${seed}" \
        --splits "$SPLITS" --out "${RES}/eval_${arm}_shuf_s${seed}.json"
  done
done

echo "ALL_DONE rc=${rc_total} $(date +%T)"
