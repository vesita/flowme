#!/usr/bin/env bash
# L3-r4 正式四臂 + 报告。用脚本文件而非 systemd-run 内联循环：
# systemd 会在 bash 之前展开 ExecStart 里的 $VAR（实测把 logs/80_${arm}_*.log 变成 logs/80__*.log）。
set -u
cd /home/vesita/coding/my/flowme
for arm in J P P+F J+GC; do
  echo "=== START arm=$arm $(date -Is) ==="
  .venv/bin/python stages/80_cardwise_vs_joint.py --arm "$arm" --seed 1234 > "logs/80_${arm}_s1234.log" 2>&1
  echo "=== END arm=$arm rc=$? $(date -Is) ==="
done
echo "=== START report $(date -Is) ==="
.venv/bin/python stages/80_cardwise_vs_joint.py --report-r4 --seed 1234 > logs/80_report_r4.log 2>&1
echo "=== END report rc=$? $(date -Is) ==="
echo "=== ALL-DONE $(date -Is) ==="
