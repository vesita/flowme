#!/usr/bin/env bash
# core_select_only 训练跑（PREREG §1/§3）。
#   $1 = joint   → 核对臂：真标签 s42/s43（T0 零误差复现 select_semantic_joint 主臂）
#   $1 = selonly → 主臂  ：真标签 s42/s43 + 随机标签 s42/s43（T1/T2/T3/T4/T5/T6）
# 由 systemd-run --user --unit=<名> 启动（会话重启不掉），绝不用 setsid nohup。
# 逐跑日志：logs/core_select_only_<arm>_s<seed>[_rand].log；结果：experiments/core_select_only/results/*.json
set -uo pipefail
cd "$(dirname "$0")/../.."

PHASE="${1:?用法：run_all.sh joint|selonly}"
PREREG="experiments/core_select_only/PREREG.md"
echo "=== PREREG mtime: $(date -r "$PREREG" '+%F %T') ==="
echo "=== run_all($PHASE) start: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑前） ==="
md5sum experiments/select_rerank/data/clean/*.jsonl

declare -a FAILED=()
run() {
  local arm="$1" seed="$2" extra="${3:-}"
  local name="${arm}_s${seed}"
  [[ -n "$extra" ]] && name="${name}_rand"
  local log="logs/core_select_only_${name}.log"
  echo "=== [$(date +%F\ %T)] START $name ==="
  uv run python experiments/core_select_only/train_core.py \
    --arm "$arm" --seed "$seed" $extra 2>&1 | tee "$log"
  local st=${PIPESTATUS[0]}
  echo "=== [$(date +%F\ %T)] $name exit=$st ==="
  if [[ $st -ne 0 ]]; then
    FAILED+=("$name")
    echo "!! $name 失败（exit=$st），继续下一个"
  fi
}

case "$PHASE" in
  joint)
    for seed in 42 43; do run joint "$seed"; done ;;
  selonly)
    for seed in 42 43; do run selonly "$seed"; done
    for seed in 42 43; do run selonly "$seed" "--randlabel"; done ;;
  *)
    echo "未知 phase：$PHASE" >&2; exit 2 ;;
esac

echo "=== run_all($PHASE) end: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑后，应与跑前一致） ==="
md5sum experiments/select_rerank/data/clean/*.jsonl
echo "=== 只读目录改动检查（应为空） ==="
find experiments/select_rerank experiments/select_pool experiments/select_semantic_joint \
     experiments/core_grad_probe experiments/core_update_probe src training tests \
     -newermt "$(date -r "$PREREG" '+%F %T')" -printf '%T+ %p\n' 2>/dev/null | head -20
echo "=== 只读目录 __pycache__ 检查（不应新增） ==="
find experiments/select_semantic_joint -name '__pycache__' -printf '%p\n' 2>/dev/null
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED RUNS: ${FAILED[*]}"
  exit 1
fi
echo "ALL DONE"
