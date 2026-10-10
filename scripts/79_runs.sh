#!/usr/bin/env bash
# L3-r3b′ 正式 6 run：A_1234 → B2_1234 → A_5678 → B2_5678 → B1_1234 → B1_5678（主判据 A vs B2 最先出）
# 每个 run 前排队等卡（排除本单元自己的进程），绝不 kill 别人的进程。
set -u
cd /home/vesita/coding/my/flowme
PY=/home/vesita/coding/my/flowme/.venv/bin/python
set -- A 1234 B2 1234 A 5678 B2 5678 B1 1234 B1 5678
while [ -n "$1" ]; do
  arm=$1; seed=$2; shift 2
  while [ -n "$(pgrep -f 'cand_cards[_v0-9]*\.py' | grep -v "^$$\$")" ]; do
    busy=0
    while systemctl --user list-units --type=service --state=running | grep -iE "flowme|dtseek" | grep -qv "flowme_l3r3bp"; do busy=1; break; done
    if [ "$busy" = "0" ]; then break; fi
    sleep 20
  done
  ph=""
  if [ "$arm" != "A" ]; then ph="--pair-hits logs/79_hits_A_s${seed}.json"; fi
  echo "=== START arm=${arm} seed=${seed} $(date -Is) ==="
  $PY stages/79_cand_cards_v2.py --arm "$arm" --seed "$seed" --k 8 --data-prefix 76 --cf $ph \
    > "logs/79_${arm}_s${seed}.log" 2>&1
  echo "=== END arm=${arm} seed=${seed} rc=$? $(date -Is) ==="
done
echo "=== ALL-DONE $(date -Is) ==="
