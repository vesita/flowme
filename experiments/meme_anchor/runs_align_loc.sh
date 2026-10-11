#!/usr/bin/env bash
# L3-m3d（PREREG_ALIGN_LOC.md）：2×2 直接对照 —— 对齐压力"加在哪一层"是不是代价的原因
#   臂 VIII：压力在**输入卡**（--align-on input），输入卡**可训**      × seed{1234,5678}
#   臂 IX  ：压力在**衔接处**（--align-on bridge），输入卡**可训**      × seed{1234,5678}
#   臂 A（对照，冻结输入 + 压力在衔接处）复用已有 run，不重跑。λ 固定 0.3。
set -u
cd /home/vesita/coding/my/flowme
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M3_STEPS:-8000}"
LAM=0.3
ck() { echo "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s$1.pt"; }

# ★① 预检（fail-closed）：输入存在性
for S in 1234 5678; do
  [ -f "$(ck "$S")" ] || { echo "[AL] 预检失败：缺 ckpt $(ck "$S")"; exit 2; }
done
[ -f stages/86g_zh_en.py ] || { echo "[AL] 预检失败：缺驱动"; exit 2; }
echo "[AL] 预检通过；start $(date -Is) λ=$LAM"

# ★② 跑后核验：每个 run 必须产出 RUN-FINAL，否则整批立刻停（不让"跑完却没结果"静默过去）
one() {  # $1=align_on $2=seed
  local ao="$1" s="$2" log="logs/86g_al_${1}_l${LAM}_s${2}.log"
  echo "[AL] ==== 压力在 $ao · 输入可训 · seed=$s start $(date -Is) ===="
  .venv/bin/python stages/86g_zh_en.py --run "$LAM" "$s" --steps "$STEPS" --eval-every 1000 \
    --maps AaBb,AbBa --loss infonce --bridge parser --align-on "$ao" \
    --init-inputs "$(ck "$s")" --init-outputs > "$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] || ! grep -q '^\[RUN-FINAL\]' "$log"; then
    echo "[AL] ✗ $ao seed=$s 失败（exit=$rc，且无 RUN-FINAL）"; tail -5 "$log"; exit 3
  fi
  echo "[AL] ✓ $ao seed=$s 完成 $(date -Is)"
}
for AO in input bridge; do
  for S in 1234 5678; do one "$AO" "$S"; done
done
echo "[AL-DONE] $(date -Is)"
