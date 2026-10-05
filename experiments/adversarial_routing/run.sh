#!/usr/bin/env bash
# 对抗集路由验证：构建对抗集 + 3-seed 探针评测（GPU，全程分钟级）
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python experiments/adversarial_routing/build_adversarial.py
uv run python experiments/adversarial_routing/run_probe.py
