#!/usr/bin/env bash
# ★ 由 systemd-run 拉起，勿直接前台跑（AGENTS.md §2：长跑用 systemd-run --user --unit=<名> --collect）
#
# L3-t1（PREREG_THINKCARD.md **v2**）：单思维卡的容量瓶颈对照（§13.2 / §13.4 的 A·B·C 臂）
#   四臂唯一变量 = 思维卡：A 旁路(none) · B 无瓶颈(mid=128) · C64 瓶颈(mid=64) · C32 紧瓶颈(mid=32)
#   其余逐字相同：--loss infonce --align-on bridge --maps AaBb,AbBa --freeze-inputs --init-outputs
#   冻结的输入卡 = v0（logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_st8000_l0.0_s{S}.pt，按 seed 配对）
#
# 运行（**默认 n=4**；PREREG v2 §1：正/负向档必须 n=4）：
#   systemd-run --user --unit=flowme-thinkcard --collect \
#     experiments/meme_anchor/runs_thinkcard.sh [SEEDS...]
#   E1 参数对齐臂（仅在档三/三′触发时）：TC_ARMS_EXTRA=1 experiments/meme_anchor/runs_thinkcard.sh <SEEDS...>
set -uo pipefail

cd /home/vesita/coding/my/flowme
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
STEPS="${TC_STEPS:-8000}"
SEEDS=("$@")
[ -z "${SEEDS[0]:-}" ] && SEEDS=(1234 5678 9012 3456)

# 臂表：名字 | bridge | mid | ff
ARMS=("A:none:64:256" "B:parser:128:256" "C64:parser:64:256" "C32:parser:32:256")
# E1（扩展臂，档三/三′触发时才跑）：mid=64 + ff=960 ⇒ 桥参 ≈298.4k，与 B 的 298.2k 差 +0.06%
[ "${TC_ARMS_EXTRA:-0}" = "1" ] && ARMS+=("E1:parser:64:960")

CK() { echo "logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_st8000_l0.0_s$1.pt"; }
LOG() { echo "logs/86g_tc_${1}_l0.3_s${2}.log"; }                  # 1=臂名 2=seed
MIDSUF() { [ "$1" = "none" ] && echo "" || echo "_$1$2"; }
FFSUF() { [ "$3" = "256" ] && echo "" || echo "_ff$3"; }
RES() { echo "logs/86g_result_infonce_AaBb_AbBa$(MIDSUF "$2" "$3")$(FFSUF "$1" "$3" "$4")_o-tf_st${STEPS}_frz_io_i-none_AaBb_AbBa_a0.5_o-tf_st8000_l0.3_s${1}.json"; }
CKT() { echo "logs/86g_ckpt_infonce_AaBb_AbBa$(MIDSUF "$2" "$3")$(FFSUF "$1" "$3" "$4")_o-tf_st${STEPS}_frz_io_i-none_AaBb_AbBa_a0.5_o-tf_st8000_l0.3_s${1}.pt"; }

echo "[TC] 预检 start $(date -Is)"
[ -f stages/86g_zh_en.py ] || { echo "[TC] 预检失败：缺驱动" >&2; exit 2; }
[ -f stages/86p_thinkcard_verdict.py ] || { echo "[TC] 预检失败：缺判据表" >&2; exit 3; }

# --- N5：R48 假结果逐档驱动真实 verdict()（★ 必改 6：以前只检查文件存在，从不调用）
"$PY" stages/86p_thinkcard_verdict.py --selfcheck >/tmp/tc_n5.log 2>&1 \
  || { echo "[TC] 预检失败：N5 selfcheck（详见 /tmp/tc_n5.log）" >&2; tail -5 /tmp/tc_n5.log >&2; exit 8; }
tail -1 /tmp/tc_n5.log

# --- N3：A 臂前向恒等（bypass == 无卡）
"$PY" stages/86p_thinkcard_verdict.py --n3-check >/tmp/tc_n3.log 2>&1 \
  || { echo "[TC] 预检失败：N3（详见 /tmp/tc_n3.log）" >&2; tail -5 /tmp/tc_n3.log >&2; exit 9; }
grep -E '^\[N3\]' /tmp/tc_n3.log | head -1

# --- N1：mid / ff 契约（128 可建、129 报错、ff=960 可建）
"$PY" - <<'PY' >/tmp/tc_n1.log 2>&1 || { echo "[TC] 预检失败：N1 mid/ff 契约（详见 /tmp/tc_n1.log）" >&2; cat /tmp/tc_n1.log >&2; exit 4; }
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
print(f"N1 OK: mid=128 桥参 {sum(p.numel() for p in p128.parameters())} / mid=64,ff=960 桥参 {sum(p.numel() for p in p960.parameters())} / mid=129 报错")
PY
cat /tmp/tc_n1.log

# --- 输入卡存在 + N6 记录 sha256
for S in "${SEEDS[@]}"; do
  [ -f "$(CK "$S")" ] || { echo "[TC] 预检失败：缺 v0 输入卡 $(CK "$S")" >&2; exit 5; }
  echo "[TC] N6 s$S init_inputs=$(CK "$S") sha256=$(sha256sum "$(CK "$S")" | cut -c1-16)（四臂按 seed 共用同一文件）"
done

# --- N4：tag 全部区分维度两两不同（bridge+mid+ff+seed）
TAGS=()
for A in "${ARMS[@]}"; do
  IFS=: read -r nm br mid ff <<<"$A"
  for S in "${SEEDS[@]}"; do TAGS+=("$(MIDSUF "$br" "$mid")$(FFSUF "$nm" "$mid" "$ff")_s$S"); done
done
uniq_n=$(printf '%s\n' "${TAGS[@]}" | sort -u | wc -l)
[ "$uniq_n" = "${#TAGS[@]}" ] || { echo "[TC] 预检失败：tag 有重复（${#TAGS[@]} 个里只有 $uniq_n 个唯一）" >&2; exit 6; }
echo "[TC] 预检通过（steps=$STEPS seeds=${SEEDS[*]} arms=${#ARMS[@]} tag 唯一 $uniq_n 个；ff/bypass 维度已在断言清单内）"
if [ "${TC_PRECHECK_ONLY:-0}" = "1" ]; then
  echo "[TC] TC_PRECHECK_ONLY=1 ⇒ 预检通过即退出（不烧 GPU）"
  exit 0
fi

for A in "${ARMS[@]}"; do
  IFS=: read -r nm br mid ff <<<"$A"
  for S in "${SEEDS[@]}"; do
    LOGF="$(LOG "$nm" "$S")"
    echo "[TC] ==== 臂 $nm (bridge=$br mid=$mid ff=$ff) seed=$S start $(date -Is) ===="
    "$PY" stages/86g_zh_en.py --run 0.3 "$S" --steps "$STEPS" --eval-every 1000 \
      --maps AaBb,AbBa --loss infonce --align-on bridge --bridge "$br" --bridge-mid "$mid" --bridge-ff "$ff" \
      --init-inputs "$(CK "$S")" --freeze-inputs --init-outputs --gen-eval-n 1000 \
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
  done
done
# --- N6（跑后真检查）：每 seed 四臂 init_inputs 同源 + 输入卡冻结
"$PY" stages/86p_thinkcard_verdict.py --n6-check --seeds "$(IFS=,; echo "${SEEDS[*]}")" || { echo "[TC] 核验失败：N6" >&2; fail=1; }
[ "$fail" = 0 ] || exit 7
echo "[TC-DONE] 臂×seed 全部完成并通过核验 $(date -Is)"
