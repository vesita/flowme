#!/usr/bin/env bash
# L3-m3b **控制臂**（PREREG_PARSER_CTRL.md）：分离"parser 机制"与"额外容量"
#   臂 IV：shared parser **随机初始化后冻结**（同尺寸、不可训）× seed 1234
#   臂 V ：**两条不共享** adapter（中/英各一张）× seed{1234,5678}
#   臂 I / II 复用已有 run，不重跑。
set -u
cd /home/vesita/coding/my/flowme
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M3_STEPS:-8000}"
V05=logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_%s.pt
echo "[CTRL] start $(date -Is)"
echo "[CTRL] ==== 臂 IV 冻结随机 parser s1234 ===="
.venv/bin/python stages/86g_zh_en.py --run 0 1234 --steps "$STEPS" --eval-every 1000 \
  --maps AaBb,AbBa --loss none --bridge parser --freeze-bridge \
  --init-inputs "$(printf $V05 1234)" --freeze-inputs --init-outputs > logs/86g_ctrl_IV_frzbr_s1234.log 2>&1
echo "[CTRL] 臂IV exit=$? $(date -Is)"
for S in 1234 5678; do
  echo "[CTRL] ==== 臂 V 不共享 adapter s$S ===="
  .venv/bin/python stages/86g_zh_en.py --run 0 "$S" --steps "$STEPS" --eval-every 1000 \
    --maps AaBb,AbBa --loss none --bridge dual \
    --init-inputs "$(printf $V05 $S)" --freeze-inputs --init-outputs > "logs/86g_ctrl_V_dual_s${S}.log" 2>&1
  echo "[CTRL] 臂V s$S exit=$? $(date -Is)"
done
echo "[CTRL-DONE] $(date -Is)"
