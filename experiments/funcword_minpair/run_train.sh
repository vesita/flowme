#!/usr/bin/env bash
# P4 三臂全量训练（2 seed）+ 随机标签负对照；由 systemd-run 单元启动。
set -u
cd /home/vesita/coding/my/DTSeek || exit 1
export PYTHONDONTWRITEBYTECODE=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONUNBUFFERED=1
mkdir -p experiments/funcword_minpair/logs
L=experiments/funcword_minpair/logs
rc=0
for s in 42 43; do
  for a in M MP "MP+"; do
    n="${a}_s${s}"
    echo "=== RUN ${n} $(date -Is)" >> "$L/train_runs.log"
    uv run python experiments/funcword_minpair/train.py --arm "$a" --seed "$s" \
      > "$L/train_${n}.log" 2>&1
    e=$?
    echo "    exit=${e} $(date -Is)" >> "$L/train_runs.log"
    [ $e -ne 0 ] && rc=$e
  done
done
# 随机标签负对照（P6）：3 臂 × 2 seed（PREREG 未写死 seed 数 ⇒ 跑满 2 seed）
for s in 42 43; do
  for a in M MP "MP+"; do
    n="${a}_s${s}_rand"
    echo "=== RUN ${n} $(date -Is)" >> "$L/train_runs.log"
    uv run python experiments/funcword_minpair/train.py --arm "$a" --seed "$s" --randlabel \
      > "$L/train_${n}.log" 2>&1
    e=$?
    echo "    exit=${e} $(date -Is)" >> "$L/train_runs.log"
    [ $e -ne 0 ] && rc=$e
  done
done
echo "rc=${rc} finished $(date -Is)" > "$L/train.done"
exit $rc
