#!/usr/bin/env bash
# L3-m3 **补跑**：只跑修复后必需的两个 run（其余三个已有效，不重复烧卡）
#   ① 臂 I  s1234（此前与臂 III 撞名被覆盖）
#   ② 臂 III（此前载错了 ckpt：用了 α=1.0 而非 α=0）
set -u
cd /home/vesita/coding/my/flowme
PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M3_STEPS:-8000}"
V05=logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s1234.pt
V00=logs/86g_ckpt_none_AaBb_l0.0_s1234.pt          # ★ α=0（纯重构），输入侧 Δ≈0

echo "[M3FIX] start $(date -Is)"
echo "[M3FIX] ==== ① 臂 I parser@a0.5 s1234 ===="
"$PY" stages/86g_zh_en.py --run 0 1234 --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss none --bridge parser --init-inputs "$V05" \
      --freeze-inputs --init-outputs > logs/86g_m3_parser_a05_s1234.log 2>&1
echo "[M3FIX] ① exit=$? $(date -Is)"
echo "[M3FIX] ==== ② 臂 III parser@a0 s1234 ===="
"$PY" stages/86g_zh_en.py --run 0 1234 --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss none --bridge parser --init-inputs "$V00" \
      --freeze-inputs --init-outputs > logs/86g_m3_parser_a00_s1234.log 2>&1
echo "[M3FIX] ② exit=$? $(date -Is)"
echo "[M3FIX-DONE] $(date -Is)"
