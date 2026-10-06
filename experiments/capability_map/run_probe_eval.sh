#!/usr/bin/env bash
# 能力筛查 第 3-5 步：等训练跑完 → 探针 → 三档评测 → 汇总
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
step() { echo "[capmap] $(date +%T) === $* ==="; }
while systemctl --user is-active dtseek-capmap-train.service >/dev/null 2>&1; do sleep 5; done
step "train 已结束：$(systemctl --user show -p Result --value dtseek-capmap-train.service)"

step "probe"
uv run python -u experiments/capability_map/probe.py > logs/capmap_probe.log 2>&1
rc=$?; echo "[capmap] probe rc=$rc"; grep -E "SELFTEST|\[probe\] [a-z]+ s|ood\]|PROBE_DONE" logs/capmap_probe.log | tail -40
[ "$rc" -ne 0 ] && { echo CAPMAP_ABORT_probe; exit 1; }

step "eval"
uv run python -u experiments/capability_map/eval_cards.py > logs/capmap_eval.log 2>&1
rc=$?; echo "[capmap] eval rc=$rc"; grep -E "ALIGN|EVAL_DONE|缺 |Traceback" logs/capmap_eval.log | tail -40

step "summarize"
uv run python -u experiments/capability_map/summarize.py > logs/capmap_summary.log 2>&1
rc=$?; echo "[capmap] summarize rc=$rc"; cat logs/capmap_summary.log
echo "CAPMAP_ALL_DONE rc=$rc $(date +%T)"
