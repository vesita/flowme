#!/usr/bin/env bash
# L3-m3 第三次补跑：**只**重跑被覆盖的"臂 I s1234"（tag 已真修，不会再与臂 III 撞名）
set -u
cd /home/vesita/coding/my/flowme
export HSA_OVERRIDE_GFX_VERSION=10.3.0
echo "[M3REFIX] start $(date -Is)"
.venv/bin/python stages/86g_zh_en.py --run 0 1234 --steps "${M3_STEPS:-8000}" --eval-every 1000 \
      --maps AaBb,AbBa --loss none --bridge parser \
      --init-inputs logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s1234.pt --freeze-inputs --init-outputs \
      > logs/86g_m3_parser_a05_s1234.log 2>&1
echo "[M3REFIX] exit=$? $(date -Is)"
