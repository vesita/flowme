#!/usr/bin/env bash
# ★ 本脚本由 systemd-run 拉起，勿直接跑
#
# H2 前置（PREREG_GENOBJ.md §1）：补 v0 输入卡 s9012 / s3456 —— 配方**逐字同 L3-m2 的 α=0.5 臂**：
#   --run 0.0 <S> --steps 8000 --eval-every 1000 --maps AaBb,AbBa --loss none --abba-weight 0.5
# 只取 ckpt（json 顺带产出）；产出后把文件名**归一成旧约定**（去掉 tag 里的 `_o-tf`），
# 好让 runs_genobj.sh 的 fail-closed 预检对 4 个 seed 用同一套文件名。
set -euo pipefail

cd /home/vesita/coding/my/flowme
export HSA_OVERRIDE_GFX_VERSION=10.3.0
PY=".venv/bin/python"

for S in 9012 3456; do
  OLD="logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s${S}.pt"
  if [ -f "$OLD" ]; then
    echo "[PREREQ] seed=$S 已有 $OLD，跳过"
    continue
  fi
  echo "[PREREQ] 训练 v0 输入卡 seed=$S start $(date -Is)"
  "$PY" stages/86g_zh_en.py --run 0.0 "$S" --steps 8000 --eval-every 1000 \
    --maps AaBb,AbBa --loss none --abba-weight 0.5 \
    > "logs/86g_alpha_a0.5_s${S}.log" 2>&1 || true

  if ! grep -q "\[RUN-FINAL\]" "logs/86g_alpha_a0.5_s${S}.log"; then
    echo "[PREREQ] seed=$S 失败：log 无 [RUN-FINAL]" >&2
    exit 7
  fi
  NEW="logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_l0.0_s${S}.pt"
  if [ ! -f "$NEW" ]; then
    echo "[PREREQ] seed=$S 失败：缺 $NEW" >&2
    exit 8
  fi
  mv "$NEW" "$OLD"
  if [ -f "logs/86g_result_none_AaBb_AbBa_a0.5_o-tf_l0.0_s${S}.json" ]; then
    mv "logs/86g_result_none_AaBb_AbBa_a0.5_o-tf_l0.0_s${S}.json" \
       "logs/86g_result_none_AaBb_AbBa_a0.5_l0.0_s${S}.json"
  fi
  echo "[PREREQ] seed=$S 完成 $(date -Is)"
done

echo "[PREREQ-DONE] v0 输入卡 4 seed 齐备 $(date -Is)"
