#!/bin/bash
# syllogism_card 全量：3 臂 × 2 seed + 随机标签对照（T / TA）
# 先排队等他人训练结束（最多 2h），绝不杀他人进程。
set -e
cd /home/vesita/coding/my/DTSeek
LOG=experiments/syllogism_card/logs/train.log
mkdir -p experiments/syllogism_card/logs
echo "=== start $(date -Is) ===" >> "$LOG"
for i in $(seq 1 720); do
  if ! pgrep -f "experiments/[a-z_]*/trai[n]" >/dev/null 2>&1; then
    echo "[queue] GPU 空闲，开跑 $(date -Is)" >> "$LOG"; break
  fi
  [ "$i" -eq 720 ] && echo "[queue] 等待超时 2h，仍开跑 $(date -Is)" >> "$LOG"
  sleep 10
done
for arm in T TA A; do
  for seed in 42 43; do
    echo "--- run arm=$arm seed=$seed $(date -Is)" >> "$LOG"
    uv run python experiments/syllogism_card/run.py --arm "$arm" --seed "$seed" >> "$LOG" 2>&1
  done
done
for arm in T TA; do
  for seed in 42 43; do
    echo "--- run arm=$arm seed=$seed randlabel $(date -Is)" >> "$LOG"
    uv run python experiments/syllogism_card/run.py --arm "$arm" --seed "$seed" --randlabel >> "$LOG" 2>&1
  done
done
echo "ALL_DONE $(date -Is)" >> "$LOG"
