"""候选选择卡族（reply_pick / cloze_fill）的单卡训练入口。

先显式 import 两张卡（各包末尾 `register(CARD)`），再原样委托
`training/train_task_card.py` —— 单卡脚本用 `resolve_tasks()` 查表，
表是在运行时由 import 建起来的，所以本入口不依赖 `builtin/__init__` 的导入顺序。

    uv run python experiments/selection_cards/run_card.py --card reply_pick \
        --samples 6000 --seed 42 --out experiments/selection_cards/reply_pick.pt
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

# 选择卡族：import 即注册（各包末尾 `register(CARD)`）
for _module in ("dtseek.tasks.builtin.reply_pick", "dtseek.tasks.builtin.cloze_fill"):
    importlib.import_module(_module)

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "training" / "train_task_card.py"

if __name__ == "__main__":
    sys.argv[0] = str(TRAINER)
    runpy.run_path(str(TRAINER), run_name="__main__")
