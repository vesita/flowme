#!/usr/bin/env bash
# L3-m1 **全量重训**（PREREG_INFONCE.md）：对比式对齐 λ_nce=0.1 × seed{1234,5678} × 10000 步。
# 基线（λ=0, 同 2 seed, 同 10000 步）**复用**主档已有 run，不在此重跑。
# 用法： systemd-run --user --unit=flowme_m1_infonce --collect bash scripts/86_runs_infonce.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M1_STEPS:-10000}"

echo "[INFONCE] start $(date -Is) steps=$STEPS pid=$$"
for SEED in 1234 5678; do
  echo "[INFONCE] ==== loss=infonce lam=0.1 seed=$SEED start $(date -Is) ===="
  "$PY" stages/86_meme_anchor.py --run 0.1 "$SEED" --steps "$STEPS" \
        --eval-every 1000 --vision-arch detr --loss infonce \
        > "logs/86_infonce_l0.1_s${SEED}.log" 2>&1
  echo "[INFONCE] seed=$SEED exit=$? $(date -Is)"
done
echo "[INFONCE-DONE] $(date -Is)"
