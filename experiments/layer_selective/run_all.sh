#!/usr/bin/env bash
# layer_selective 训练跑（PREREG §2/§3，跑前写死）。
#   $1 = smoke → ALL 臂 100 步探跑（与 core_select_only 的 selonly_selftest_s42 对账，冒烟）
#   $1 = real  → 五臂 × 2 seed 真标签（F/ALL/EMB/TOP/ADPT × 42/43）
#   $1 = rand  → 五臂 × 2 seed 随机标签（X4 门禁）
# 由 systemd-run --user --unit=<名> 启动（会话重启不掉），绝不用 setsid nohup。
# 逐跑日志：logs/layer_selective_<arm>_s<seed>[_rand].log；结果：experiments/layer_selective/results/*.json
set -uo pipefail
cd "$(dirname "$0")/../.."

PHASE="${1:?用法：run_all.sh smoke|real|rand}"
PREREG="experiments/layer_selective/PREREG.md"
echo "=== PREREG mtime: $(date -r "$PREREG" '+%F %T') ==="
echo "=== run_all($PHASE) start: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑前） ==="
md5sum experiments/select_rerank/data/clean/*.jsonl

declare -a FAILED=()
run() {
  local arm="$1" seed="$2" extra="${3:-}" steps="${4:-}"
  local name="${arm}_s${seed}"
  [[ -n "$extra" ]] && name="${name}_rand"
  local log="logs/layer_selective_${name}.log"
  local stepargs=()
  [[ -n "$steps" ]] && stepargs=(--steps "$steps" --tag "probe${steps}")
  echo "=== [$(date +%F\ %T)] START $name ==="
  uv run python experiments/layer_selective/train_ls.py \
    --arm "$arm" --seed "$seed" $extra "${stepargs[@]}" 2>&1 | tee "$log"
  local st=${PIPESTATUS[0]}
  echo "=== [$(date +%F\ %T)] $name exit=$st ==="
  if [[ $st -ne 0 ]]; then
    FAILED+=("$name")
    echo "!! $name 失败（exit=$st），继续下一个"
  fi
}

case "$PHASE" in
  smoke)
    run ALL 42 "" 100 ;;
  real)
    for arm in F ALL EMB TOP ADPT; do for seed in 42 43; do run "$arm" "$seed"; done; done ;;
  rand)
    for arm in F ALL EMB TOP ADPT; do for seed in 42 43; do run "$arm" "$seed" "--randlabel"; done; done ;;
  *)
    echo "未知 phase：$PHASE" >&2; exit 2 ;;
esac

echo "=== run_all($PHASE) end: $(date '+%F %T') ==="
echo "=== 只读数据指纹（跑后，应与跑前一致） ==="
md5sum experiments/select_rerank/data/clean/*.jsonl
echo "=== 只读目录改动检查（应为空） ==="
find experiments/select_rerank experiments/select_pool experiments/select_semantic_joint \
     experiments/core_select_only experiments/capability_map src training tests \
     -newermt "$(date -r "$PREREG" '+%F %T')" -printf '%T+ %p\n' 2>/dev/null | head -20
echo "=== 只读目录 __pycache__ 检查（不应新增） ==="
find experiments/select_semantic_joint experiments/core_select_only -name '__pycache__' -printf '%p\n' 2>/dev/null
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED RUNS: ${FAILED[*]}"
  exit 1
fi
echo "ALL DONE"
