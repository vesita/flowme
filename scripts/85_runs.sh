#!/usr/bin/env bash
# S85 · L3-r8「预训练后逐卡微调 vs 联合微调」四臂 × 2 seed 的**顺序**队列（L3_PLAN §20）。
#   臂：J（联合 3000 步，对照）/ JT（预训练 2000 → 联合微调 1000）/
#       JT-P（预训练 → 只微调 选择卡+重整卡）/ JT-P1（预训练 → 只微调 重整卡）
#   跑序：每 seed **先 JT → JT-P → JT-P1**（R8-b 主判据最早可判），再 J（R8-a/R8-c 对照）。
#   ★单卡 8G：**一个 GPU 队列，绝不并发**；起卡前查在跑的 GPU 单元，撞上则排队等（后起者让位，绝不 kill 别人）。
#   ★所有带变量的展开都在**本脚本**里发生：systemd-run 只调这份脚本，绝不把循环内联进 systemd-run
#     （systemd 会在 bash 之前展开 $VAR —— r4/r6 实测踩过，文件名会被吃掉）。
#   ★日志**按臂分文件**：logs/85_<ARM>_s<seed>.log；判据汇总 logs/85_report_r8{,_s<seed>}.log。
#   用法：bash scripts/85_runs.sh          # 正式（2 seed × 4 臂 + 2 次 --report-r8）
#         MODE=smoke bash scripts/85_runs.sh   # 探路（50 步，不作判据）
set -u
REPO=/home/vesita/coding/my/flowme
cd "$REPO" || exit 1
PY=$REPO/.venv/bin/python
DRV=$REPO/logs/85_runs_driver.log
MODE="${MODE:-formal}"
SEEDS="${SEEDS:-1234 5678}"
SMOKE_FLAG=""
[ "$MODE" = "smoke" ] && SMOKE_FLAG="--smoke"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONUNBUFFERED=1
mkdir -p "$REPO/logs"
exec > >(tee -a "$DRV") 2>&1

log() { echo "[$(date -Is)] $*"; }

others_running() {
  systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null \
    | grep -iE "flowme|dtseek" | awk '{print $1}' | grep -v '^flowme_l3r8\.service$' | head -1
}
wait_for_others() {
  local who
  who="$(others_running || true)"
  while [ -n "$who" ]; do
    log "★排队等 $who（别人的单元在跑；绝不 kill 对方）"
    sleep 20
    who="$(others_running || true)"
  done
}
gpu_procs() { ps -eo pid,etime,cmd | grep -E "\.venv/bin/python .*stages/" | grep -v grep || true; }

peak_of() { grep -h '^\[MEM\]' "$1" 2>/dev/null | tail -1 | grep -o '★峰值分配=[0-9]*' | cut -d= -f2; }

run_arm() {                       # $1=arm $2=seed $3=slug
  local arm=$1 seed=$2 slug=$3
  local log="$REPO/logs/85_${slug}_s${seed}${MODE_SUFFIX}.log"
  wait_for_others
  log "START arm=$arm seed=$seed ${SMOKE_FLAG:-} → $(basename "$log")"
  # shellcheck disable=SC2086
  $PY stages/85_pretrain_then_cardwise.py --arm "$arm" --seed "$seed" $SMOKE_FLAG > "$log" 2>&1
  local rc=$?
  log "END   arm=$arm seed=$seed rc=$rc 峰值=$(peak_of "$log")MiB :: $(grep -h '^\[META\]' "$log" | tail -1)"
  return $rc
}

MODE_SUFFIX=""
[ "$MODE" = "smoke" ] && MODE_SUFFIX="_smoke"

log "=== S85 L3-r8 队列开始（MODE=$MODE seeds=$SEEDS） ==="
log "起手现场：别的 GPU 单元=$(systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null | grep -icE 'flowme|dtseek') 个；GPU 进程："
gpu_procs | sed 's/^/    /'
wait_for_others

# ---- 仪器自检：相位 1 的 A/A 噪声地板 + 与臂标签无关性（同进程三条轨迹，~3 min） ----
PRE_STEPS=200
[ "$MODE" = "smoke" ] && PRE_STEPS=40
wait_for_others
log "START --precheck --precheck-steps $PRE_STEPS（相位 1 的 A/A 地板）"
$PY stages/85_pretrain_then_cardwise.py --precheck --precheck-steps "$PRE_STEPS" --seed 1234 \
    > "$REPO/logs/85_precheck.log" 2>&1
log "END   --precheck rc=$? :: $(grep -h '噪声地板' "$REPO/logs/85_precheck.log" | tail -1)"

for seed in $SEEDS; do
  log "---- seed=$seed（先 JT/JT-P/JT-P1 让 R8-b 先可判，再 J）----"
  run_arm JT    "$seed" JT
  run_arm JT-P  "$seed" JTP
  run_arm JT-P1 "$seed" JTP1
  run_arm J     "$seed" J
  wait_for_others
  log "START --report-r8 --seeds $seed"
  $PY stages/85_pretrain_then_cardwise.py --report-r8 --seeds "$seed" $SMOKE_FLAG \
      > "$REPO/logs/85_report_r8_s${seed}${MODE_SUFFIX}.log" 2>&1
  log "END   report seed=$seed rc=$?"
done

SEEDS_CSV="$(echo $SEEDS | tr ' ' ',')"
wait_for_others
log "START --report-r8 --seeds $SEEDS_CSV（合并）"
$PY stages/85_pretrain_then_cardwise.py --report-r8 --seeds "$SEEDS_CSV" $SMOKE_FLAG \
    > "$REPO/logs/85_report_r8${MODE_SUFFIX}.log" 2>&1
log "END   合并 report rc=$?"
log "=== S85 队列全部结束 $(date -Is) ==="
exit 0
