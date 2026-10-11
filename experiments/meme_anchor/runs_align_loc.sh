#!/usr/bin/env bash
# L3-m3d（PREREG_ALIGN_LOC.md）：2×2 直接对照 —— 对齐压力"加在哪一层"是不是代价的原因
#   臂 VIII：压力在**输入卡**（--align-on input），输入卡**可训**      × seed{1234,5678}
#   臂 IX  ：压力在**衔接处**（默认 bridge），输入卡**可训**            × seed{1234,5678}
#   臂 A（对照，冻结输入 + 压力在衔接处）复用已有 run，不重跑。
#   λ 固定 0.3（两臂同 λ，只差"压力层"与"输入是否可训"）。
set -u
cd /home/vesita/coding/my/flowme
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M3_STEPS:-8000}"
LAM=0.3
ck() { echo "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s$1.pt"; }

# ★ fail-closed 预检
for S in 1234 5678; do
  [ -f "$(ck "$S")" ] || { echo "[AL] 预检失败：缺 ckpt $(ck "$S")"; exit 2; }
done
[ -f stages/86g_zh_en.py ] || { echo "[AL] 预检失败：缺驱动"; exit 2; }
echo "[AL] 预检通过；start $(date -Is) λ=$LAM"

for AO in input bridge; do           # input = 臂 VIII · bridge = 臂 IX
  for S in 1234 5678; do
    echo "[AL] ==== 压力在 $AO · 输入可训 · seed=$S start $(date -Is) ===="
    .venv/bin/python stages/86g_zh_en.py --run "$LAM" "$S" --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss infonce --bridge parser --align-on "$AO" \
      --init-inputs "$(ck "$S")" --init-outputs \
      > "logs/86g_al_${AO}_l${LAM}_s${S}.log" 2>&1
    echo "[AL] $AO seed=$S exit=$? $(date -Is)"
  done
done
echo "[AL-DONE] $(date -Is)"
