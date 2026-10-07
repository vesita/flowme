#!/usr/bin/env bash
# bag_modules 补跑：B5 实测 M2(题元) 覆盖率 0.085 < 0.10 ⇒ 按 PREREG §4「该臂用到的模块
# 全部可用」，只有 M1(类型)+M3(槽型) 的臂才可能判「有效」⇒ 补跑 U(−role) × 2 seed + 分析。
# （补跑决定在看到主表之后，理由是跑前写死的 B5 门槛；门槛本身不放宽，见 REPORT。）
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EXP=experiments/bag_modules
LOG=$EXP/logs
mkdir -p "$LOG"

for s in 42 43; do
  echo "=== U type,cls s$s $(date -Is) ==="
  uv run python "$EXP/train.py" --arm U --seed "$s" --mods type,cls 2>&1 | tee "$LOG/U_s${s}_mtype-cls.log"
done

echo "=== analyze $(date -Is) ==="
uv run python "$EXP/analyze.py" 2>&1 | tee "$LOG/analyze.log"
echo "=== EXTRA DONE $(date -Is) ==="
