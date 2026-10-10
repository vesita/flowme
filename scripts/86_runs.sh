#!/usr/bin/env bash
# L3-m1 模因锚定试验 · 夜间跑批：6 run 串行（λ ∈ {0, 0.5, 2.0} × seed ∈ {1234, 5678}）+ 汇总判词
#
# 规格与判据的唯一真相源：experiments/meme_anchor/PREREG.md
# 用法： systemd-run --user --unit=flowme_m1 --collect bash scripts/86_runs.sh
#   ★ 循环写在本脚本文件里，不写进 systemd ExecStart —— systemd 会先展开 ExecStart 里的 $VAR
#     （L3_PLAN.md §12.5(d) 踩过：内联 for + ${arm} 会得到空变量）。
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M1_STEPS:-10000}"
ARCH="${M1_ARCH:-detr}"

echo "[RUNNER] start $(date -Is) steps=$STEPS arch=$ARCH pid=$$"
echo "[RUNNER] gpu: $(nvidia-smi 2>/dev/null | head -1 || echo 'rocm (无 rocm-smi)')"

for cfg in 0:1234 0:5678 0.5:1234 0.5:5678 2:1234 2:5678; do
  LAM="${cfg%%:*}"
  SEED="${cfg##*:}"
  echo "[RUNNER] ==== lam=$LAM seed=$SEED start $(date -Is) ===="
  "$PY" stages/86_meme_anchor.py --run "$LAM" "$SEED" --steps "$STEPS" \
        --eval-every 1000 --vision-arch "$ARCH" \
        > "logs/86_run_l${LAM}_s${SEED}.log" 2>&1
  rc=$?
  echo "[RUNNER] lam=$LAM seed=$SEED exit=$rc $(date -Is)"
done

echo "[RUNNER] ==== verdict $(date -Is) ===="
"$PY" stages/86_meme_anchor.py --verdict > logs/86_verdict.log 2>&1
echo "[RUNNER] verdict exit=$?"
echo "[ALL-DONE] m1 $(date -Is)"
