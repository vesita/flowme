#!/usr/bin/env bash
# core_update_probe 一次性测量跑（由 systemd-run --user --unit=<名> 启动，绝不用 setsid nohup）。
#   用法：bash experiments/core_update_probe/run_probe.sh [smoke]
#     无参    → 正式跑：step 1/50/200（200 步），AdamW 实际更新口径
#     smoke   → 冒烟  ：只测 step 1，2 步就停（产物另存，不污染正式结果）
# 日志：logs/core_update_probe_s42.log / logs/core_update_probe_smoke.log
# 产物：experiments/core_update_probe/results/*.json
set -uo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-}"
TAG="s42"
PYARGS=(--seed 42 --steps 200)
if [[ "$MODE" == "smoke" ]]; then
  TAG="smoke"
  PYARGS=(--seed 42 --smoke)
fi
LOG="logs/core_update_probe_${TAG}.log"

echo "=== core_update_probe(${TAG}) start: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑前） ==="
md5sum experiments/select_rerank/data/clean/*.jsonl
uv run python experiments/core_update_probe/probe_update.py "${PYARGS[@]}" 2>&1 | tee "$LOG"
st=${PIPESTATUS[0]}
echo "=== core_update_probe(${TAG}) end: $(date '+%F %T') exit=$st ==="
echo "=== 只读目录改动检查（应为空；以 PREREG mtime 为界） ==="
find experiments/select_semantic_joint experiments/core_grad_probe experiments/select_pool \
  experiments/select_rerank src training \
  -newermt "$(date -r experiments/core_update_probe/PREREG.md '+%F %T')" \
  -printf '%T+ %p\n' 2>/dev/null | head -20
exit "$st"
