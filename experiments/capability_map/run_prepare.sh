#!/usr/bin/env bash
# 能力筛查 第 1 步：构建/缓存 5 份数据集 + 导出 held-out 划分（有明确终点）
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
echo "[prepare] $(date +%T) start"
uv run python -u experiments/capability_map/prepare.py
rc=$?
echo "[prepare] $(date +%T) rc=${rc}"
exit $rc
