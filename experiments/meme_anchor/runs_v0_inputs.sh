#!/usr/bin/env bash
# ★ 由 systemd-run 拉起，勿直接前台跑（AGENTS.md §2：长跑用 systemd-run --user --unit=<名> --collect）
#
# v0 输入卡（α=0.5 档，L3-m2 配方）—— **本机（noxy / RTX 5060 / CUDA）重训**
#   4 seed × 8000 步：--maps AaBb,AbBa --loss none --abba-weight 0.5（逐字同 L3-m2 的 α=0.5 臂）
#   产物：logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_st8000_l0.0_s{S}.pt
#         logs/86g_result_none_AaBb_AbBa_a0.5_o-tf_st8000_l0.0_s{S}.json
#
# 运行：
#   systemd-run --user --unit=flowme-v0inputs --collect experiments/meme_anchor/runs_v0_inputs.sh
set -uo pipefail

cd /home/vesita/coding/my/flowme
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
STEPS="${V0_STEPS:-8000}"
SEEDS=(1234 5678 9012 3456)

echo "[V0IN] 预检 start $(date -Is)"
[ -f stages/86g_zh_en.py ] || { echo "[V0IN] 预检失败：缺驱动" >&2; exit 2; }
[ -f logs/86_tokenizer/char_tokenizer.json ] || { echo "[V0IN] 预检失败：缺字符分词器" >&2; exit 3; }
[ -s logs/86_cache_zhen/pairs.jsonl ] || { echo "[V0IN] 预检失败：缺中英配对缓存" >&2; exit 4; }
echo "[V0IN] 预检通过（steps=$STEPS seeds=${SEEDS[*]}）"

for S in "${SEEDS[@]}"; do
  LOG="logs/86g_alpha_a0.5_s${S}.log"
  echo "[V0IN] ==== seed=$S start $(date -Is) ===="
  "$PY" stages/86g_zh_en.py --run 0.0 "$S" --steps "$STEPS" --eval-every 1000 \
    --maps AaBb,AbBa --loss none --abba-weight 0.5 --gen-eval-n 1000 > "$LOG" 2>&1
  rc=$?
  echo "[V0IN] seed=$S exit=$rc $(date -Is)"
  [ "$rc" = 0 ] || echo "[V0IN] seed=$S 非零退出（继续下一个，收尾统一核验）" >&2
done

echo "[V0IN] 跑后核验"
fail=0
for S in "${SEEDS[@]}"; do
  LOG="logs/86g_alpha_a0.5_s${S}.log"
  CK="logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_st${STEPS}_l0.0_s${S}.pt"
  JS="logs/86g_result_none_AaBb_AbBa_a0.5_o-tf_st${STEPS}_l0.0_s${S}.json"
  for f in "$LOG" "$CK" "$JS"; do
    [ -e "$f" ] || { echo "[V0IN] 核验失败：缺 $f" >&2; fail=1; }
  done
  grep -q '\[RUN-FINAL\]' "$LOG" || { echo "[V0IN] 核验失败：$LOG 无 [RUN-FINAL]" >&2; fail=1; }
done
[ "$fail" = 0 ] || exit 5
echo "[V0IN-DONE] 4 个输入卡 run 完成并通过核验 $(date -Is)"
