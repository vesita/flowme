#!/usr/bin/env bash
# L3-m3c（PREREG_BRIDGE_ALIGN.md）：**在衔接处施加对齐压力**（对齐项作用于适配卡输出，输入卡冻结）
#   臂 VI : --bridge parser --loss infonce --lam 0.1 × seed{1234,5678}
#   臂 VII: --bridge parser --loss infonce --lam 0.3 × seed{1234,5678}
#   臂 I（对照，结构相同但 loss=none）复用已有 run，不重跑。
set -u
cd /home/vesita/coding/my/flowme
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M3_STEPS:-8000}"
ck() { echo "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s$1.pt"; }

# ★ fail-closed 预检（今天第四个坑的教训：路径没对上要在第一秒暴露）
for S in 1234 5678; do
  [ -f "$(ck "$S")" ] || { echo "[BA] 预检失败：缺 ckpt $(ck "$S")"; exit 2; }
done
[ -f stages/86g_zh_en.py ] || { echo "[BA] 预检失败：缺驱动"; exit 2; }
echo "[BA] 预检通过；start $(date -Is)"

for LAM in 0.1 0.3; do
  for S in 1234 5678; do
    echo "[BA] ==== 衔接处 infonce λ=$LAM seed=$S start $(date -Is) ===="
    .venv/bin/python stages/86g_zh_en.py --run "$LAM" "$S" --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss infonce --bridge parser \
      --init-inputs "$(ck "$S")" --freeze-inputs --init-outputs \
      > "logs/86g_ba_l${LAM}_s${S}.log" 2>&1
    echo "[BA] λ=$LAM seed=$S exit=$? $(date -Is)"
  done
done
echo "[BA-DONE] $(date -Is)"
