#!/usr/bin/env bash
# select_pool 全部训练跑（PREREG §2/§3）：5 臂 × 2 seed（真标签）+ 5 臂 × seed42（随机标签）= 15 跑。
# 由 systemd-run --user --unit=<名> 启动（会话重启不掉）。
# 逐跑日志：logs/select_pool_<arm>_s<seed>[_rand].log；结果：experiments/select_pool/results/*.json
# 单跑失败不中断后续（每跑独立落盘），最后汇总 exit code。
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "=== PREREG mtime: $(date -r experiments/select_pool/PREREG.md '+%F %T') ==="
echo "=== run_all start: $(date '+%F %T') ==="

declare -a FAILED=()
run() {
  local arm="$1" seed="$2" extra="${3:-}"
  local name="${arm}_s${seed}"
  [[ -n "$extra" ]] && name="${name}_rand"
  local log="logs/select_pool_${name}.log"
  echo "=== [$(date +%F\ %T)] START $name ==="
  uv run python experiments/select_pool/train_pool.py \
    --arm "$arm" --seed "$seed" $extra 2>&1 | tee "$log"
  local st=${PIPESTATUS[0]}
  echo "=== [$(date +%F\ %T)] $name exit=$st ==="
  if [[ $st -ne 0 ]]; then
    FAILED+=("$name")
    echo "!! $name 失败（exit=$st），继续下一个"
  fi
}

# 真标签：5 臂 × 2 seed
for arm in A B C D Aplus; do
  for seed in 42 43; do
    run "$arm" "$seed"
  done
done
# Q3 随机标签对照：5 臂 × seed42
for arm in A B C D Aplus; do
  run "$arm" 42 "--randlabel"
done

echo "=== run_all end: $(date '+%F %T') ==="
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED RUNS: ${FAILED[*]}"
  exit 1
fi
echo "ALL DONE"
