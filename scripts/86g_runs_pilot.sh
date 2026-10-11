#!/usr/bin/env bash
# L3-m2 **先导对比**（3000 步 · seed 1234）：只 AaBb（零假设）vs AaBb+AbBa（用户的映射方案）
#   用法： systemd-run --user --unit=flowme_m2_pilot --collect bash scripts/86g_runs_pilot.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M2_STEPS:-3000}"

echo "[PILOT] start $(date -Is) steps=$STEPS pid=$$"
echo "[PILOT] ---- ① 只 AaBb（自映射；两路之间没有任何联系 ⇒ 期望 Δ≈0）----"
"$PY" stages/86g_zh_en.py --run 0 1234 --steps "$STEPS" --eval-every 1000 \
      --maps AaBb --loss none > logs/86g_pilot_AaBb_only.log 2>&1
echo "[PILOT] ① exit=$? $(date -Is)"
echo "[PILOT] ---- ② AaBb+AbBa（自映射 + 互映射/翻译）----"
"$PY" stages/86g_zh_en.py --run 0 1234 --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss none > logs/86g_pilot_AaBb_AbBa.log 2>&1
echo "[PILOT] ② exit=$? $(date -Is)"
echo "[PILOT-DONE] $(date -Is)"
