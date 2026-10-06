#!/usr/bin/env bash
# cumulative_add：连续加卡（每步温启动联合，口径 = core_keep 的 J3）全流程串行。
# 只写 experiments/cumulative_add/ 与 logs/cumadd_*；不碰 src/ training/ tests/ dev-notes/
# 以及 experiments/{core_keep,capability_map,core_branch,additivity,out_invariants,input_type,core_generalize}/
#
# 顺序（PREREG §4.1 要求：bands 必须在链上开跑之前落盘）：
#   0) 步 0 门禁 × 2 seed
#   1) 对照（单加卡）× 2 卡 × 2 seed
#   2) bands.py → bands.json
#   3) 链上 step1 → step2 → step3 × 2 seed（每步起点门禁）
#   4) summarize.py → results.json
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
CARDS=experiments/cumulative_add/cards
mkdir -p "$CARDS" logs
rc_total=0
declare -A OK

run() {
  local name="$1"; shift
  local log="logs/cumadd_${name}.log"
  echo "[cumadd] $(date +%T) start ${name} -> ${log}"
  "$@" > "$log" 2>&1
  local r=$?
  echo "[cumadd] $(date +%T) done  ${name} rc=${r}"
  grep -E "SELFTEST_GATE .* verdict|SELFTEST_3 verdict|SELFTEST_OPT 冻结|CUMADD_METRICS|Traceback|SystemExit|GATE_FAIL" "$log" | tail -8
  OK["$name"]=$r
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

train() {  # train <name> <args...>
  local name="$1"; shift
  run "$name" uv run python -u experiments/cumulative_add/train_cumulative.py "$@" \
      --out "$CARDS/${name}.pt" --metrics "$CARDS/${name}.json"
}

# ---------------- -1) 预检：排队等别的 dtseek-* 单元把单卡 GPU 让出来（绝不杀别人的进程） ----------------
SELF_UNIT="dtseek-cumadd.service"
echo "[cumadd] preflight: 排队等待其它 dtseek-* 单元结束（最多 4 小时）..."
for i in $(seq 1 240); do
  busy=$(systemctl --user list-units --plain --no-legend 'dtseek-*' 2>/dev/null \
         | awk -v me="$SELF_UNIT" '$3=="active" && $1!=me {print $1}' | tr '\n' ' ')
  # 兜底：任何在跑 DTSeek 实验脚本的 python（不看它挂在哪个单元下）
  strays=$(pgrep -af "[.]venv/bin/python -u experiments/" | awk '{print $1}' | tr '\n' ' ')
  if [ -z "${busy// /}" ] && [ -z "${strays// /}" ]; then
    echo "[cumadd] GPU 空闲，开始 $(date +%T)"
    break
  fi
  echo "[cumadd] $(date +%T) 排队中（忙）: units=[$busy] pids=[$strays]"
  sleep 60
done

# ---------------- 0) 步 0 门禁（G0：四卡基线 vs capability_map 记录值，逐位） ----------------
for s in 42 43; do
  train "gate0_s${s}" --mode gate --seed "$s"
done

# ---------------- 1) 对照：单加卡（P2 参照）----------------
for s in 42 43; do
  [ "${OK[gate0_s${s}]:-1}" -ne 0 ] && { echo "[cumadd] gate0_s${s} 失败，跳过该 seed 的对照"; continue; }
  for c in idiom ownership; do
    train "ctrl_${c}_s${s}" --mode control --card "$c" --seed "$s" \
          --prev-metrics "$CARDS/gate0_s${s}.json"
  done
done

# ---------------- 2) 噪声带（链上开跑之前落盘） ----------------
run "bands" uv run python -u experiments/cumulative_add/bands.py
if [ "${OK[bands]:-1}" -ne 0 ]; then
  echo "CUMADD_ALL_ABORT bands 失败 ⇒ 链上不许开跑 rc=${rc_total}"
  exit 1
fi

# ---------------- 3) 链上三步 ----------------
for s in 42 43; do
  [ "${OK[gate0_s${s}]:-1}" -ne 0 ] && { echo "[cumadd] gate0_s${s} 失败，跳过链上 s${s}"; continue; }
  train "step1_s${s}" --mode chain --step 1 --seed "$s" \
        --prev-metrics "$CARDS/gate0_s${s}.json"
  [ "${OK[step1_s${s}]:-1}" -ne 0 ] && { echo "[cumadd] step1_s${s} 失败，跳过 step2/3 s${s}"; continue; }
  train "step2_s${s}" --mode chain --step 2 --seed "$s" \
        --init "$CARDS/step1_s${s}.pt" --prev-metrics "$CARDS/step1_s${s}.json"
  [ "${OK[step2_s${s}]:-1}" -ne 0 ] && { echo "[cumadd] step2_s${s} 失败，跳过 step3 s${s}"; continue; }
  train "step3_s${s}" --mode chain --step 3 --seed "$s" \
        --init "$CARDS/step2_s${s}.pt" --prev-metrics "$CARDS/step2_s${s}.json"
done

# ---------------- 4) 汇总判定 ----------------
run "summarize" uv run python -u experiments/cumulative_add/summarize.py

echo "CUMADD_ALL_DONE rc=${rc_total} $(date +%T)"
exit $rc_total
