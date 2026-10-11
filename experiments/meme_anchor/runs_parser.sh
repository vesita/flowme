#!/usr/bin/env bash
# L3-m3 **parser 卡试验**（判据见 experiments/meme_anchor/PREREG_PARSER.md）
#   臂 I  主    ：parser(mid=64) + 冻结 α=0.5 的输入卡（四卡都从 v0 载入，输出卡继续训）× seed{1234,5678}
#   臂 II 对照  ：无 parser（恒等）+ 同一冻结输入 × seed{1234,5678}   ← 只差"有没有 parser"
#   臂 III 强测 ：parser + 冻结 **α=0（纯重构）** 的输入卡 × seed 1234 ← 输入侧 Δ≈0 时能否从零造出对齐
# ★ 常规位置是 scripts/86g_runs_parser.sh，但该路径被另一 DSH 会话（在做**另一个仓库**的 Rust 工作）
#   以 `scripts/` 前缀独占（collab_lock 冲突、等待超时）⇒ 按纪律不抢占，放本实验目录内；锁释放后可原样移回。
# 用法： systemd-run --user --unit=flowme_m3_parser --collect bash experiments/meme_anchor/runs_parser.sh
set -u
cd /home/vesita/coding/my/flowme

PY=.venv/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0
STEPS="${M3_STEPS:-8000}"

echo "[M3] start $(date -Is) steps=$STEPS pid=$$"
run() {  # $1=日志标签 $2=bridge $3=载入的 ckpt $4=seed
  echo "[M3] ==== $1 bridge=$2 seed=$4 start $(date -Is) ===="
  "$PY" stages/86g_zh_en.py --run 0 "$4" --steps "$STEPS" --eval-every 1000 \
        --maps AaBb,AbBa --loss none --bridge "$2" \
        --init-inputs "$3" --freeze-inputs --init-outputs \
        > "logs/86g_m3_$1_s$4.log" 2>&1
  echo "[M3] $1 seed=$4 exit=$? $(date -Is)"
}

for S in 1234 5678; do
  run parser_a05 parser "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s${S}.pt" "$S"
done
for S in 1234 5678; do
  run noparser_a05 none "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s${S}.pt" "$S"
done
# 臂 I s1234 重跑：早先它与臂 III 撞名被覆盖（tag 已修，不会再撞）
run parser_a05 parser "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s1234.pt" 1234
# ★ 臂 III 要的是 **α=0（纯重构）** 的 v0：文件名是 `..._none_AaBb_l0.0_s1234.pt`
#   （早先误写成 `..._none_AaBb_AbBa_l0.0_s1234.pt` = α=1.0，输入侧 Δ=+0.1917 而不是 ≈0 —— 已修）
run parser_a00 parser "logs/86g_ckpt_none_AaBb_l0.0_s1234.pt" 1234
echo "[M3-DONE] $(date -Is)"
