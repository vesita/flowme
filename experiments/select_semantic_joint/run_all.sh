#!/usr/bin/env bash
# select_semantic_joint 训练跑（PREREG §2/§3）。
#   $1 = frozen → 核对臂：真标签 s42/s43 + 随机标签 s42/s43（R0/R3）
#   $1 = joint  → 主臂  ：真标签 s42/s43 + 随机标签 s42/s43（R1/R2/R3/R4/R5/R6）
# 由 systemd-run --user --unit=<名> 启动（会话重启不掉）。
# 逐跑日志：logs/select_semantic_joint_<arm>_s<seed>[_rand].log；结果：experiments/select_semantic_joint/results/*.json
set -uo pipefail
cd "$(dirname "$0")/../.."

PHASE="${1:?用法：run_all.sh frozen|joint}"
echo "=== PREREG mtime: $(date -r experiments/select_semantic_joint/PREREG.md '+%F %T') ==="
echo "=== run_all($PHASE) start: $(date '+%F %T') ==="
echo "=== 只读数据指纹 ==="
md5sum experiments/select_rerank/data/clean/*.jsonl

declare -a FAILED=()
run() {
  local arm="$1" seed="$2" extra="${3:-}"
  local name="${arm}_s${seed}"
  [[ -n "$extra" ]] && name="${name}_rand"
  local log="logs/select_semantic_joint_${name}.log"
  echo "=== [$(date +%F\ %T)] START $name ==="
  uv run python experiments/select_semantic_joint/train_sem.py \
    --arm "$arm" --seed "$seed" $extra 2>&1 | tee "$log"
  local st=${PIPESTATUS[0]}
  echo "=== [$(date +%F\ %T)] $name exit=$st ==="
  if [[ $st -ne 0 ]]; then
    FAILED+=("$name")
    echo "!! $name 失败（exit=$st），继续下一个"
  fi
}

case "$PHASE" in
  frozen)
    for seed in 42 43; do run frozen "$seed"; done
    for seed in 42 43; do run frozen "$seed" "--randlabel"; done
    ;;
  joint)
    for seed in 42 43; do run joint "$seed"; done
    for seed in 42 43; do run joint "$seed" "--randlabel"; done
    ;;
  *)
    echo "未知 phase：$PHASE" >&2; exit 2 ;;
esac

echo "=== run_all($PHASE) end: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑后） ==="
md5sum experiments/select_rerank/data/clean/*.jsonl
echo "=== 只读目录改动检查（应为空） ==="
find experiments/select_rerank experiments/select_pool -newermt "$(date -r experiments/select_semantic_joint/PREREG.md '+%F %T')" \
  -printf '%T+ %p\n' 2>/dev/null | head -20
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED RUNS: ${FAILED[*]}"
  exit 1
fi
echo "ALL DONE"
