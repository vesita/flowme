#!/usr/bin/env bash
# S84 · L3-r7（L3_PLAN §19）推理专用探针驱动：**先 fp32 主档，再 fp16 独立档**，最后出 R7-a..e。
#   ★一个 GPU 队列，绝不并发；fp32 / fp16 是**两列**，各自独立落盘、不混比（§19 写死）。
#   ★所有带变量的展开都在**本脚本**里发生：systemd-run 只调这份脚本，绝不把循环内联进 systemd-run
#     （systemd 会在 bash 之前展开 $VAR —— r4/r6 都实测踩过，文件名会被吃掉）。
#   ★日志**按档分文件**：logs/84_<dtype>.log；判据汇总 logs/84_report_r7.log。
#   ★起卡前先查在跑的 GPU 单元/进程，撞上则**排队等**（后起者让位，绝不 kill 别人的进程）。
set -u
REPO=/home/vesita/coding/my/flowme
cd "$REPO" || exit 1
PY=$REPO/.venv/bin/python
DRV=$REPO/logs/84_runs_driver.log
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONUNBUFFERED=1
mkdir -p "$REPO/logs"
exec > >(tee -a "$DRV") 2>&1

log() { echo "[$(date -Is)] $*"; }

others_running() {
  systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null \
    | grep -iE "flowme|dtseek" | awk '{print $1}' | grep -v '^flowme_l3r7\.service$' | head -1
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

log "=== S84 L3-r7 开始 $(date -Is) ==="
log "起手现场：别的 GPU 单元=$(systemctl --user list-units --type=service --state=running --no-legend 2>/dev/null | grep -icE 'flowme|dtseek') 个；GPU 进程："
gpu_procs | sed 's/^/    /'
wait_for_others

for dt in fp32 fp16; do
  wait_for_others
  log "START dtype=$dt → logs/84_${dt}.log"
  $PY stages/84_infer_probe.py --dtype "$dt" > "$REPO/logs/84_${dt}.log" 2>&1
  rc=$?
  log "END   dtype=$dt rc=$rc :: $(grep -h '^\[META\]' "$REPO/logs/84_${dt}.log" | tail -1)"
done

wait_for_others
log "START --report-r7"
$PY stages/84_infer_probe.py --report-r7 > "$REPO/logs/84_report_r7.log" 2>&1
log "END   --report-r7 rc=$?"
log "=== S84 队列全部结束 $(date -Is) ==="
exit 0
