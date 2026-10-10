#!/usr/bin/env bash
# 79 三臂 smoke（各 50 步，真训）+ 复评复核（--load --cf）
set -u
cd /home/vesita/coding/my/flowme
PY=/home/vesita/coding/my/flowme/.venv/bin/python
# 等卡：只等别人的 cand_cards 进程（排除本脚本自身）；并等 systemd 单元（排除本单元）
while pgrep -f 'cand_cards[_v0-9]*\.py' | grep -qv "^$$\$"; do
  if ! systemctl --user list-units --type=service --state=running | grep -iE "flowme|dtseek" | grep -qv "flowme_l3r3bp"; then break; fi
  sleep 20
done
for arm in A B1 B2; do
  echo "=== START smoke-train arm=${arm} $(date -Is) ==="
  $PY stages/79_cand_cards_v2.py --arm "$arm" --seed 1234 --k 8 --data-prefix 76 --smoke --cf \
    > "logs/79_smoke_${arm}.log" 2>&1
  echo "=== END smoke-train arm=${arm} rc=$? $(date -Is) ==="
done
for arm in A B1 B2; do
  echo "=== START smoke-recheck arm=${arm} $(date -Is) ==="
  $PY stages/79_cand_cards_v2.py --arm "$arm" --seed 1234 --k 8 --data-prefix 76 --smoke --cf \
    --load "logs/79_ckpt_${arm}_k8_s1234_smoke.pt" \
    > "logs/79_smoke_${arm}_recheck.log" 2>&1
  echo "=== END smoke-recheck arm=${arm} rc=$? $(date -Is) ==="
done
echo "=== SMOKE-DONE $(date -Is) ==="
