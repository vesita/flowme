#!/usr/bin/env bash
# L3-m1 链接单元：等主跑批（flowme_m1）结束后**自动接上补救轮**，不让 GPU 空转。
# 用法： systemd-run --user --unit=flowme_m1_chain --collect bash scripts/86_chain.sh
set -u
cd /home/vesita/coding/my/flowme

echo "[CHAIN] 等待 flowme_m1 结束 $(date -Is)"
while systemctl --user is-active --quiet flowme_m1; do sleep 20; done
echo "[CHAIN] flowme_m1 已结束 $(date -Is) ⇒ 起补救轮"
bash scripts/86_runs_remedy.sh
echo "[CHAIN-DONE] $(date -Is)"
