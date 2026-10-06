#!/usr/bin/env bash
# select_rerank 全部训练跑（PREREG §2/§3）：2 臂 × 2 seed × {真标签, 随机标签} = 8 跑。
# 由 systemd-run --user --unit=dtseek-select-rerank 启动（会话重启不掉）。
# 逐跑日志：logs/select_rerank_<arm>_s<seed>[_rand].log；结果：experiments/select_rerank/results/*.json
set -uo pipefail
cd "$(dirname "$0")/../.."

for spec in \
    "clean:42:" "clean:43:" "clean:42:--randlabel" "clean:43:--randlabel" \
    "shortcut:42:" "shortcut:43:" "shortcut:42:--randlabel" "shortcut:43:--randlabel"; do
  IFS=: read -r arm seed extra <<<"$spec"
  name="${arm}_s${seed}"
  [[ -n "$extra" ]] && name="${name}_rand"
  log="logs/select_rerank_${name}.log"
  echo "=== [$(date +%F\ %T)] START $name ==="
  PYTHONUNBUFFERED=1 uv run python experiments/select_rerank/train_rerank.py \
    --arm "$arm" --seed "$seed" $extra 2>&1 | tee "$log"
  st=${PIPESTATUS[0]}
  echo "=== [$(date +%F\ %T)] $name exit=$st ==="
  if [[ $st -ne 0 ]]; then
    echo "ABORT: $name 失败（exit=$st），后续跳过"
    exit "$st"
  fi
done
echo "ALL DONE"
