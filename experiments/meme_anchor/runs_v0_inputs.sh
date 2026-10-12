#!/usr/bin/env bash
# ★ 由 systemd-run 拉起，勿直接前台跑（AGENTS.md §2：长跑用 systemd-run --user --unit=<名> --collect）
#
# v0（α=0.5 档，L3-m2 配方）输入卡重训 —— **本机（noxy / RTX 5060 / CUDA）**
#   4 seed × 8000 步：--maps AaBb,AbBa --loss none --abba-weight 0.5 [--slots N]
#   n=16（默认）：logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_st8000_l0.0_s{S}.pt
#   n≠16        ：logs/86g_ckpt_none_AaBb_AbBa_a0.5_n{N}_o-tf_st8000_l0.0_s{S}.pt
# ★★ 产物名**必须**带 `_n{N}`（N≠16 时）—— 否则会用 n=4 的同名文件**覆盖 n=16 的 v0 输入卡**
#     （PREREG_SLOTN §1 前置门 P2；复核实锤 ①）。脚本末尾按 result JSON 的 `n_slots` 复核。
#
# 运行：
#   systemd-run --user --unit=flowme-v0inputs --collect experiments/meme_anchor/runs_v0_inputs.sh
#   V0_SLOTS=4 V0_UNIT_SUFFIX=n4 systemd-run --user --unit=flowme-v0inputs-n4 --collect experiments/meme_anchor/runs_v0_inputs.sh
set -uo pipefail

cd /home/vesita/coding/my/flowme
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
STEPS="${V0_STEPS:-8000}"
SLOTS="${V0_SLOTS:-16}"
SEEDS=(1234 5678 9012 3456)
NSUF=""; [ "$SLOTS" = "16" ] || NSUF="_n${SLOTS}"

echo "[V0IN] 预检 start $(date -Is)（slots=$SLOTS steps=$STEPS）"
[ -f stages/86g_zh_en.py ] || { echo "[V0IN] 预检失败：缺驱动" >&2; exit 2; }
[ -f logs/86_tokenizer/char_tokenizer.json ] || { echo "[V0IN] 预检失败：缺字符分词器" >&2; exit 3; }
[ -s logs/86_cache_zhen/pairs.jsonl ] || { echo "[V0IN] 预检失败：缺中英配对缓存" >&2; exit 4; }
"$PY" stages/86g_zh_en.py --help 2>&1 | grep -q -- '--slots' || { echo "[V0IN] 预检失败：驱动没有 --slots（前置门 P1 未过）" >&2; exit 6; }
# 产物名守卫：n≠16 时目标名必须真的不同名（P2）
if [ "$SLOTS" != "16" ]; then
  case "$NSUF" in "_n$SLOTS") ;; *) echo "[V0IN] 预检失败：NSUF 异常（$NSUF）" >&2; exit 7;; esac
fi
echo "[V0IN] 预检通过（seeds=${SEEDS[*]} 产物后缀='${NSUF}'）"

for S in "${SEEDS[@]}"; do
  LOG="logs/86g_alpha_a0.5${NSUF}_s${S}.log"
  echo "[V0IN] ==== seed=$S start $(date -Is) ===="
  "$PY" stages/86g_zh_en.py --run 0.0 "$S" --steps "$STEPS" --eval-every 1000 \
    --maps AaBb,AbBa --loss none --abba-weight 0.5 --slots "$SLOTS" --gen-eval-n 1000 > "$LOG" 2>&1
  rc=$?
  echo "[V0IN] seed=$S exit=$rc $(date -Is)"
  [ "$rc" = 0 ] || echo "[V0IN] seed=$S 非零退出（继续下一个，收尾统一核验）" >&2
done

echo "[V0IN] 跑后核验"
fail=0
for S in "${SEEDS[@]}"; do
  LOG="logs/86g_alpha_a0.5${NSUF}_s${S}.log"
  TAG="none_AaBb_AbBa_a0.5${NSUF}_o-tf_st${STEPS}"
  CK="logs/86g_ckpt_${TAG}_l0.0_s${S}.pt"
  JS="logs/86g_result_${TAG}_l0.0_s${S}.json"
  for f in "$LOG" "$CK" "$JS"; do
    [ -e "$f" ] || { echo "[V0IN] 核验失败：缺 $f" >&2; fail=1; }
  done
  grep -q '\[RUN-FINAL\]' "$LOG" || { echo "[V0IN] 核验失败：$LOG 无 [RUN-FINAL]" >&2; fail=1; }
  [ -e "$JS" ] && "$PY" -c "
import json,sys
d=json.load(open('$JS'))
assert int(d.get('n_slots',16))==int('$SLOTS'), f\"n_slots 不符：{d.get('n_slots')} != $SLOTS\"
" || { echo "[V0IN] 核验失败：$JS 的 n_slots 不是 $SLOTS" >&2; fail=1; }
done
[ "$fail" = 0 ] || exit 5
echo "[V0IN-DONE] 4 个输入卡 run 完成并通过核验（slots=$SLOTS）$(date -Is)"
