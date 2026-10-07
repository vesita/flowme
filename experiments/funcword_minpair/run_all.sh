#!/usr/bin/env bash
# P4 一键复现：PREREG → 构造 → 自检 → 12 次训练 → 12 次评测 → 汇总表 → verdict
# 全部经 systemd-run 启动（绝不用 setsid nohup &）；每步有明确终点。
set -u
cd /home/vesita/coding/my/DTSeek || exit 1
export PYTHONDONTWRITEBYTECODE=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONUNBUFFERED=1
E=experiments/funcword_minpair
L=$E/logs
mkdir -p "$L"

step() { echo "### $*  $(date -Is)"; }

step "0 PREREG 已在位（mtime 必须早于首次训练）"
test -f "$E/PREREG.md" || { echo "缺 PREREG.md"; exit 1; }

step "1 构造最小对 + 分离断言"
uv run python "$E/build_data.py" > "$L/build_data.log" 2>&1 || exit 1
grep -E "^\[sep\]|^\[rules\]|^\[cover\]|^\[build\]|^\[heldout\]" "$L/build_data.log"

step "2 模型自检（核冻结 / 三臂初值逐位同 / M loss == struct B / 构造 + 分离）"
uv run python "$E/model.py" --selfcheck > "$L/selfcheck.log" 2>&1 || exit 1
grep "selfcheck" "$L/selfcheck.log"

step "3 冒烟（60 步 × 3 臂，全通才继续）"
for a in M MP "MP+"; do
  uv run python "$E/train.py" --arm "$a" --seed 42 --steps 60 > "$L/smoke_${a}.log" 2>&1 \
    || { echo "冒烟失败: $a"; exit 1; }
done
echo "冒烟三臂全通"

step "4 全量训练 12 次"
rm -f "$L/train.done"
systemd-run --user --unit=dtseek-fwmp-train --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash "$E/run_train.sh" || exit 1
for i in $(seq 1 300); do [ -f "$L/train.done" ] && break; sleep 5; done
cat "$L/train.done" || { echo "训练未结束"; exit 1; }

step "5 评测 12 次"
rm -f "$L/eval.done"
systemd-run --user --unit=dtseek-fwmp-eval --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash "$E/run_eval.sh" || exit 1
for i in $(seq 1 300); do [ -f "$L/eval.done" ] && break; sleep 5; done
cat "$L/eval.done" || { echo "评测未结束"; exit 1; }

step "6 汇总"
uv run python "$E/analyze.py" > "$L/analyze.log" 2>&1 || exit 1
tail -5 "$L/analyze.log"

step "完成"
