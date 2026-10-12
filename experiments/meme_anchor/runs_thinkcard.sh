#!/usr/bin/env bash
# ★ 由 systemd-run 拉起，勿直接前台跑（AGENTS.md §2：长跑用 systemd-run --user --unit=<名> --collect）
#
# L3-t1 / L3-t2：单思维卡的容量瓶颈对照（A 旁路 · B 无瓶颈 128 · C64 瓶颈 64 · C32 紧瓶颈 32）
#   L3-t1（PREREG_THINKCARD v2）：--slots 16（默认）
#   L3-t2（PREREG_SLOTN）：      TC_SLOTS=4（或 8）—— 产物 tag 自动带 `_n{N}`
#   其余逐字相同：--loss infonce --align-on bridge --maps AaBb,AbBa --freeze-inputs --init-outputs
#
# 运行：
#   systemd-run --user --unit=flowme-thinkcard  --collect experiments/meme_anchor/runs_thinkcard.sh            # n=16
#   TC_SLOTS=4 systemd-run --user --unit=flowme-slotn --collect experiments/meme_anchor/runs_thinkcard.sh     # n=4
#   扩展臂 E1（档三/三′）：TC_ARMS_EXTRA=1 ...
set -uo pipefail

cd /home/vesita/coding/my/flowme
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
STEPS="${TC_STEPS:-8000}"
SLOTS="${TC_SLOTS:-16}"
SEEDS=("$@")
[ -z "${SEEDS[0]:-}" ] && SEEDS=(1234 5678 9012 3456)
NSUF=""; [ "$SLOTS" = "16" ] || NSUF="_n${SLOTS}"

# 臂表：名字 | bridge | mid | ff
ARMS=("A:none:64:256" "B:parser:128:256" "C64:parser:64:256" "C32:parser:32:256")
# E1（扩展臂，档三/三′触发时才跑）：mid=64 + ff=960 ⇒ 桥参 ≈298.4k，与 B 的 298.2k 差 +0.06%
[ "${TC_ARMS_EXTRA:-0}" = "1" ] && ARMS+=("E1:parser:64:960")

# 冻结的输入卡（按 seed 配对；n≠16 时带 `_n{N}`）——与 runs_v0_inputs.sh 的产物名逐字一致
CK() { echo "logs/86g_ckpt_none_AaBb_AbBa_a0.5${NSUF}_o-tf_st8000_l0.0_s$1.pt"; }
LOG() { echo "logs/86g_tc${NSUF}_${1}_l0.3_s${2}.log"; }                  # 1=臂名 2=seed
MIDSUF() { [ "$1" = "none" ] && echo "" || echo "_$1$2"; }
FFSUF() { [ "$3" = "256" ] && echo "" || echo "_ff$3"; }
# tag 顺序（与 86g.build_tag 逐字对齐）：loss_maps[_a{α}][_n{N}][_parser{mid}][_ff{ff}][_o-{obj}][_st{steps}][_frz][_io][_i-{init}]
TAIL() { echo "${NSUF}$(MIDSUF "$2" "$3")$(FFSUF "$1" "$3" "$4")_o-tf_st${STEPS}_frz_io_i-none_AaBb_AbBa_a0.5${NSUF}_o-tf_st8000"; }
RES() { echo "logs/86g_result_infonce_AaBb_AbBa$(TAIL "$1" "$2" "$3" "$4")_l0.3_s${1}.json"; }
CKT() { echo "logs/86g_ckpt_infonce_AaBb_AbBa$(TAIL "$1" "$2" "$3" "$4")_l0.3_s${1}.pt"; }

echo "[TC] 预检 start $(date -Is)（slots=$SLOTS steps=$STEPS seeds=${SEEDS[*]}）"
[ -f stages/86g_zh_en.py ] || { echo "[TC] 预检失败：缺驱动" >&2; exit 2; }
[ -f stages/86p_thinkcard_verdict.py ] || { echo "[TC] 预检失败：缺判据表" >&2; exit 3; }

# --- N9：槽数契约（默认 16 逐位不变；16 槽 ckpt 配 4 槽必须非零退出；tag 金标）★ 前置门 P1
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest tests/test_slots_config.py -q >/tmp/tc_n9.log 2>&1 \
  || { echo "[TC] 预检失败：N9 槽数契约（详见 /tmp/tc_n9.log）" >&2; tail -5 /tmp/tc_n9.log >&2; exit 10; }
tail -1 /tmp/tc_n9.log

# --- N5：R48 假结果逐档驱动真实 verdict()
"$PY" stages/86p_thinkcard_verdict.py --selfcheck >/tmp/tc_n5.log 2>&1 \
  || { echo "[TC] 预检失败：N5 selfcheck（详见 /tmp/tc_n5.log）" >&2; tail -5 /tmp/tc_n5.log >&2; exit 8; }
tail -1 /tmp/tc_n5.log

# --- N3：A 臂前向恒等（bypass == 无卡）
"$PY" stages/86p_thinkcard_verdict.py --n3-check >/tmp/tc_n3.log 2>&1 \
  || { echo "[TC] 预检失败：N3（详见 /tmp/tc_n3.log）" >&2; tail -5 /tmp/tc_n3.log >&2; exit 9; }
grep -E '^\[N3\]' /tmp/tc_n3.log | head -1

# --- N1：mid / ff / slots 契约
"$PY" - <<PY >/tmp/tc_n1.log 2>&1 || { echo "[TC] 预检失败：N1 mid/ff/slots 契约（详见 /tmp/tc_n1.log）" >&2; cat /tmp/tc_n1.log >&2; exit 4; }
import importlib.util, sys
from pathlib import Path
root = Path("/home/vesita/coding/my/flowme")
sys.path.insert(0, str(root / "experiments" / "meme_anchor"))       # model_zh_en 依赖同目录的 model.py
spec = importlib.util.spec_from_file_location("mz", root / "experiments/meme_anchor/model_zh_en.py")
mz = importlib.util.module_from_spec(spec); spec.loader.exec_module(mz)
p128 = mz.ParserCard(mid=128)
p960 = mz.ParserCard(mid=64, ff=960)
try:
    mz.ParserCard(mid=129); print("N1 FAIL: mid=129 没有报错"); sys.exit(1)
except AssertionError:
    pass
n = int("$SLOTS")
m = mz.build(1234, False, "parser", 64, 256, n_slots=n)
assert m.n_slots == n and m.zh_in(   __import__("torch").zeros(2, 7, dtype=__import__("torch").long)).shape == (2, n, 128), "slots 契约不符"
print(f"N1 OK: mid=128 桥参 {sum(p.numel() for p in p128.parameters())} / mid=64,ff=960 桥参 {sum(p.numel() for p in p960.parameters())} / mid=129 报错 / n={n} 模因形状 (2,{n},128) / 默认 16 逐位不变（由 N9 覆盖）")
PY
cat /tmp/tc_n1.log

# --- 输入卡存在 + N6 记录 sha256（n 维已在路径里）
for S in "${SEEDS[@]}"; do
  [ -f "$(CK "$S")" ] || { echo "[TC] 预检失败：缺输入卡 $(CK "$S")（前置门 P5 未过）" >&2; exit 5; }
  echo "[TC] N6 s$S init_inputs=$(CK "$S") sha256=$(sha256sum "$(CK "$S")" | cut -c1-16)"
done

# --- N4：tag 全部区分维度两两不同（bridge+mid+ff+**n**+seed）
TAGS=()
for A in "${ARMS[@]}"; do
  IFS=: read -r nm br mid ff <<<"$A"
  for S in "${SEEDS[@]}"; do TAGS+=("${NSUF}$(MIDSUF "$br" "$mid")$(FFSUF "$nm" "$mid" "$ff")_s$S"); done
done
uniq_n=$(printf '%s\n' "${TAGS[@]}" | sort -u | wc -l)
[ "$uniq_n" = "${#TAGS[@]}" ] || { echo "[TC] 预检失败：tag 有重复（${#TAGS[@]} 个里只有 $uniq_n 个唯一）" >&2; exit 6; }
# 负向用例：n≠16 时，目标 ckpt 名**必须**与 n=16 的同臂同名产物不同（复核实锤 ①/⑦）
if [ "$SLOTS" != "16" ]; then
  for A in "${ARMS[@]}"; do
    IFS=: read -r nm br mid ff <<<"$A"
    NSUF="" ; BASE="$(CKT "${SEEDS[0]}" "$br" "$mid" "$ff")" ; NSUF="_n${SLOTS}"
    DIFF="$(CKT "${SEEDS[0]}" "$br" "$mid" "$ff")"
    [ "$BASE" != "$DIFF" ] || { echo "[TC] 预检失败：n=$SLOTS 的产物名与 n=16 相同（会覆盖）" >&2; exit 11; }
  done
fi
echo "[TC] 预检通过（arms=${#ARMS[@]} tag 唯一 $uniq_n 个；ff/bypass/n 维度已在断言清单内）"
if [ "${TC_PRECHECK_ONLY:-0}" = "1" ]; then
  echo "[TC] TC_PRECHECK_ONLY=1 ⇒ 预检通过即退出（不烧 GPU）"
  exit 0
fi

for A in "${ARMS[@]}"; do
  IFS=: read -r nm br mid ff <<<"$A"
  for S in "${SEEDS[@]}"; do
    LOGF="$(LOG "$nm" "$S")"
    echo "[TC] ==== 臂 $nm (bridge=$br mid=$mid ff=$ff n=$SLOTS) seed=$S start $(date -Is) ===="
    "$PY" stages/86g_zh_en.py --run 0.3 "$S" --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss infonce --align-on bridge --bridge "$br" --bridge-mid "$mid" --bridge-ff "$ff" \
      --slots "$SLOTS" --init-inputs "$(CK "$S")" --freeze-inputs --init-outputs --gen-eval-n 1000 \
      > "$LOGF" 2>&1
    rc=$?
    echo "[TC] 臂 $nm seed=$S exit=$rc $(date -Is)"
    [ "$rc" = 0 ] || echo "[TC] 臂 $nm seed=$S 非零退出（继续，收尾统一核验）" >&2
  done
done

echo "[TC] 跑后核验"
fail=0
for A in "${ARMS[@]}"; do
  IFS=: read -r nm br mid ff <<<"$A"
  for S in "${SEEDS[@]}"; do
    LOGF="$(LOG "$nm" "$S")"; RESF="$(RES "$S" "$br" "$mid" "$ff")"; CKF="$(CKT "$S" "$br" "$mid" "$ff")"
    for f in "$LOGF" "$RESF" "$CKF"; do
      [ -e "$f" ] || { echo "[TC] 核验失败：缺 $f" >&2; fail=1; }
    done
    grep -q '\[RUN-FINAL\]' "$LOGF" || { echo "[TC] 核验失败：$LOGF 无 [RUN-FINAL]" >&2; fail=1; }
    [ -e "$RESF" ] && "$PY" -c "
import json
d=json.load(open('$RESF'))
assert int(d.get('n_slots',16))==int('$SLOTS'), f\"n_slots 不符：{d.get('n_slots')} != $SLOTS\"
" || { echo "[TC] 核验失败：$RESF 的 n_slots 不是 $SLOTS" >&2; fail=1; }
  done
done
"$PY" stages/86p_thinkcard_verdict.py --n6-check --seeds "$(IFS=,; echo "${SEEDS[*]}")" || { echo "[TC] 核验失败：N6" >&2; fail=1; }
[ "$fail" = 0 ] || exit 7
echo "[TC-DONE] 臂×seed 全部完成并通过核验（slots=$SLOTS）$(date -Is)"
