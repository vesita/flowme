#!/usr/bin/env bash
# S82 · L3-r6（L3_PLAN §15）AMP 四臂 × 2 seed 的**顺序**队列（单卡 8G：绝不并发）
#   臂：J-FP32 / J-AMP-bf16 / J-AMP-fp16 / J+GC-AMP（较优 AMP 档，由 --best-amp 依据读数选）
#   ★所有带变量的展开都在**本脚本**里发生：systemd-run 只调这份脚本，绝不把循环内联进 systemd-run
#     （systemd 会在 bash 之前展开 $VAR，实测把 logs/82_${arm}_*.log 变成 logs/82__*.log）。
#   ★日志**按臂分文件**（同名 `>` 会覆盖）：logs/82_<arm>_s<seed>.log。
#   ★起卡前先查在跑的 GPU 单元/进程，**排队等**（撞上则后起者让位；绝不 kill 别人的进程）。
#   ★每 run 之前与之后都复查一次：别的单元起来了就让位（等待，不抢）。
set -u
REPO=/home/vesita/coding/my/flowme
cd "$REPO" || exit 1
PY=$REPO/.venv/bin/python
DRV=$REPO/logs/82_runs_driver.log
SEEDS="${SEEDS:-1234 5678}"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONUNBUFFERED=1
mkdir -p "$REPO/logs"
exec > >(tee -a "$DRV") 2>&1

log() { echo "[$(date -Is)] $*"; }

others_running() {
  systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null \
    | grep -iE "flowme|dtseek" | awk '{print $1}' | grep -v '^flowme_l3r6\.service$' | head -1
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

peak_of()   { grep -o '★峰值=[0-9]*' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }
usable_of() { grep -h '^\[META\]' "$1" 2>/dev/null | tail -1 | grep -o 'usable=[A-Za-z]*' | cut -d= -f2; }

run_arm() {                       # $1=arm $2=amp_dtype（只有 J+GC-AMP 用）$3=seed
  local arm=$1 dt=$2 seed=$3
  local log="$REPO/logs/82_${arm}_s${seed}.log"
  local extra=""
  [ "$arm" = "J+GC-AMP" ] && extra="--amp-dtype $dt"
  wait_for_others
  log "START arm=$arm amp_dtype=${extra:-—} seed=$seed → logs/82_${arm}_s${seed}.log"
  # shellcheck disable=SC2086
  $PY stages/82_amp_probe.py --arm "$arm" --seed "$seed" $extra > "$log" 2>&1
  local rc=$?
  log "END   arm=$arm seed=$seed rc=$rc 峰值=$(peak_of "$log")MiB usable=$(usable_of "$log") :: $(grep -h '^\[META\]' "$log" | tail -1)"
  return $rc
}

log "=== S82 L3-r6 队列开始（seeds=$SEEDS） ==="
log "起手现场：GPU 单元=$(systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null | grep -icE 'flowme|dtseek') 个；GPU 进程："
gpu_procs | sed 's/^/    /'
wait_for_others

for seed in $SEEDS; do
  log "---- seed=$seed ----"
  run_arm J-FP32      -    "$seed"
  run_arm J-AMP-bf16  -    "$seed"
  run_arm J-AMP-fp16  -    "$seed"
  BEST="$($PY stages/82_amp_probe.py --best-amp --seed "$seed" 2>&1 | tee -a "$DRV" | tail -1)"
  log "较优 AMP 档（--best-amp，可用性优先 → 峰值最低 → 同峰值取每步更快）= $BEST"
  run_arm J+GC-AMP "$BEST" "$seed"
done

# ---- 叠加性 / 开关诊断（都是实测，非四臂判据） ----
FIRST_SEED="$(echo $SEEDS | awk '{print $1}')"
BEST="$($PY stages/82_amp_probe.py --best-amp --seed "$FIRST_SEED" 2>&1 | tail -1)"
for seed in $SEEDS; do
  wait_for_others
  log "START 叠加性实测 --probe-cross-amp amp_dtype=$BEST seed=$seed"
  $PY stages/82_amp_probe.py --probe-cross-amp --amp-dtype "$BEST" --seed "$seed" \
      > "$REPO/logs/82_probecross_amp_s${seed}.log" 2>&1
  log "END   叠加性 rc=$? :: $(grep -h '★叠加性' "$REPO/logs/82_probecross_amp_s${seed}.log" | tail -1)"
  wait_for_others
  log "START SDPA 开关诊断 amp_dtype=$BEST seed=$seed"
  $PY stages/82_amp_probe.py --sdp-flag-probe --amp-dtype "$BEST" --seed "$seed" \
      > "$REPO/logs/82_sdpflag_s${seed}.log" 2>&1
  log "END   SDPA 诊断 rc=$? :: $(grep -h '★SDPA 开关的显存效应' "$REPO/logs/82_sdpflag_s${seed}.log" | tail -1)"
done

# ---- 判据（CPU） ----
log "START --report-r6 分 seed"
for seed in $SEEDS; do
  $PY stages/82_amp_probe.py --report-r6 --seeds "$seed" > "$REPO/logs/82_report_r6_s${seed}.log" 2>&1
  log "END   report seed=$seed rc=$?"
done
SEEDS_CSV="$(echo $SEEDS | tr ' ' ',')"
$PY stages/82_amp_probe.py --report-r6 --seeds "$SEEDS_CSV" > "$REPO/logs/82_report_r6.log" 2>&1
log "END   report 合并 rc=$? → logs/82_report_r6.log"

log "=== S82 队列全部结束 ==="
exit 0
