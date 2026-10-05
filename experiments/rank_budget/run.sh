#!/bin/bash
set -euo pipefail

# 进入项目根目录
cd "$(dirname "$0")/../.."

echo "=== 开始 DTSeek 任务卡容量与秩预算探针实验 (E1-E4) ==="

# 1. 检查或拆分独立任务卡 (checkpoints/cards)
if [ ! -d "checkpoints/cards" ] || [ ! -f "checkpoints/cards/pronoun.pt" ]; then
    echo ">> 任务卡目录 checkpoints/cards 不存在，正在从 multitask_v2_dtseek.pt 拆分..."
    uv run python scripts/split_checkpoint.py --ckpt checkpoints/multitask_v2_dtseek.pt
fi

# 2. 运行单次主测量 (seed 42)
echo ">> [E1] 测量基座表征层级有效秩与利用率..."
uv run python experiments/rank_budget/measure_e1.py --seed 42 --out experiments/rank_budget/e1_results.json

echo ">> [E2] 测量任务卡解码器有效秩分布..."
uv run python experiments/rank_budget/measure_e2.py --out experiments/rank_budget/e2_results.json

echo ">> [E3] 测量卡片间任务梯度余弦与子空间冲突..."
uv run python experiments/rank_budget/measure_e3.py --seed 42 --out experiments/rank_budget/e3_results.json

echo ">> [E4] 测量任务路由线性探针可分性上界与独立负对照..."
uv run python experiments/rank_budget/measure_e4.py --seed 42 --out experiments/rank_budget/e4_results.json

# 3. 运行多 seed (42, 43, 44) 复核与聚合生成 results.json
echo ">> 运行多 seed (42, 43, 44) 稳定性复核并汇总 results.json..."
uv run python experiments/rank_budget/run_sweep.py

echo "=== 实验全部完成！结果已存入 experiments/rank_budget/results.json ==="
