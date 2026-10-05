"""5 卡联合训练薄 wrapper：参数与 A 臂 CLI 完全一致，只多测墙钟与峰值显存。

等价 CLI（判据跑前写死，见 dev-notes/12 §5）：
    uv run python training/train_multitask.py \\
        --tasks pronoun sentiment relation person negation \\
        --epochs 16 --seed 42 \\
        --ckpt checkpoints/multitask_v5_negation_dtseek.pt \\
        --metrics checkpoints/multitask_v5_negation_metrics.json

与 A 臂的唯一差别：任务列表多 negation、产物换名（不覆盖 multitask_v2_*）。
峰值显存必须从真实训练进程取 `torch.cuda.max_memory_allocated()`，所以不走子进程。
"""
from __future__ import annotations

import json
import sys
import time

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

import torch  # noqa: E402
from train_multitask import train_multitask  # noqa: E402

TASKS = ["pronoun", "sentiment", "relation", "person", "negation"]
CKPT = "checkpoints/multitask_v5_negation_dtseek.pt"
METRICS = "checkpoints/multitask_v5_negation_metrics.json"
STAT = "experiments/negation_card/run_5card_stat.json"


def main() -> int:
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[WRAPPER] start {started} tasks={TASKS}", flush=True)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_multitask(
        num_epochs=16,
        batch_size=64,
        lr_base=3e-4,
        lr_head=1e-3,
        samples_per_task=6000,
        task_samples=None,
        tasks=TASKS,
        steps_per_epoch=None,
        freeze_base=False,
        freeze_after_warmup=None,
        init_from=None,
        seed=42,
        ckpt_path=CKPT,
        metrics_path=METRICS,
    )
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    payload = {
        "tasks": TASKS,
        "epochs": 16,
        "seed": 42,
        "batch_size": 64,
        "wall_seconds": round(wall, 1),
        "peak_mem_bytes": peak,
        "peak_mem_gib": round(peak / 2**30, 2),
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "start": started,
        "end": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    Path(STAT).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WRAPPER] wall={wall:.1f}s peak_mem={peak / 2**30:.2f}GiB cuda={cuda}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
