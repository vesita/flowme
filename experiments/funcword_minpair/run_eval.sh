#!/usr/bin/env bash
# P4 评测批次：12 个模型（3 臂 × 2 seed × {真标签, 随机标签}）
set -u
cd /home/vesita/coding/my/DTSeek || exit 1
export PYTHONDONTWRITEBYTECODE=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONUNBUFFERED=1
L=experiments/funcword_minpair/logs
mkdir -p "$L"
rc=0
for s in 42 43; do
  for a in M MP "MP+"; do
    for extra in "" "--randlabel"; do
      if [ -z "$extra" ]; then n="${a}_s${s}"; else n="${a}_s${s}_rand"; fi
      echo "=== EVAL ${n} $(date -Is)" >> "$L/eval_runs.log"
      uv run python experiments/funcword_minpair/eval_arms.py --arm "$a" --seed "$s" $extra \
        > "$L/eval_${n}.log" 2>&1
      e=$?
      echo "    exit=${e} $(date -Is)" >> "$L/eval_runs.log"
      [ $e -ne 0 ] && rc=$e
    done
  done
done
echo "rc=${rc} finished $(date -Is)" > "$L/eval.done"
exit $rc
