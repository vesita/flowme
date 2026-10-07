#!/usr/bin/env bash
# core_probe 探针跑（判据见 PREREG.md，mtime 早于任何一次运行）。
#   用法：bash experiments/core_probe/run.sh reconcile|p0|smoke|full
# 由 systemd-run --user --unit=<名> 启动（会话重启不掉），绝不用 setsid nohup。
# 日志：experiments/core_probe/logs/<phase>.log；产物：experiments/core_probe/results/*.json
# 只读：checkpoints/、experiments/*/cache|data、src/、training/；只写本目录。
set -uo pipefail
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0

PHASE="${1:?用法：run.sh reconcile|p0|smoke|full}"
PREREG="experiments/core_probe/PREREG.md"
HERE="experiments/core_probe"
mkdir -p "$HERE/logs"

echo "=== PREREG mtime: $(date -r "$PREREG" '+%F %T') ==="
echo "=== run.sh($PHASE) start: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑前） ==="
md5sum experiments/core_keep/cache/sentiment_32000.pkl \
       experiments/core_keep/cache/person_6000.pkl \
       experiments/two_channel_head/data/train.jsonl \
       experiments/two_channel_head/data/test.jsonl
echo "=== GPU / 占用（跑前） ==="
ps -eo pid,etime,cmd | grep -E "train_|run_all|uv run python" | grep -v grep | head
systemctl --user list-units 'dtseek*' --no-legend

case "$PHASE" in
  reconcile) uv run python "$HERE/reconcile.py" 2>&1 | tee "$HERE/logs/reconcile.log" ;;
  p0)       uv run python "$HERE/run_probe.py" --phase p0 2>&1 | tee "$HERE/logs/p0.log" ;;
  smoke)    uv run python "$HERE/run_probe.py" --phase full --smoke --tag _smoke \
              ${PROBE_DEVICE:+--device "$PROBE_DEVICE"} 2>&1 | tee "$HERE/logs/smoke.log" ;;
  full)     uv run python "$HERE/run_probe.py" --phase full --tag "" \
              ${PROBE_DEVICE:+--device "$PROBE_DEVICE"} 2>&1 | tee "$HERE/logs/full.log" ;;
  *) echo "未知 phase：$PHASE" >&2; exit 2 ;;
esac
st=${PIPESTATUS[0]}
echo "=== run.sh($PHASE) end: $(date '+%F %T') exit=$st ==="
echo "=== 只读数据指纹（跑后，应与跑前一致） ==="
md5sum experiments/core_keep/cache/sentiment_32000.pkl \
       experiments/core_keep/cache/person_6000.pkl \
       experiments/two_channel_head/data/train.jsonl \
       experiments/two_channel_head/data/test.jsonl
echo "=== 只读目录改动检查（应为空） ==="
find experiments/core_keep experiments/two_channel_head checkpoints src training \
     -newermt "$(date -r "$PREREG" '+%F %T')" -printf '%T+ %p\n' 2>/dev/null | head -20
exit $st
