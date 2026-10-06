"""锚定候选打标卡（anchored_sel）—— 任务声明 + 数据入口。

只在**本实验进程内**注册（`run_card.py` import 后再委托 `training/train_task_card.py`），
不改 `src/`、不改注册表源文件：注册表是运行时 dict，import 即注册。

数据来自 `build_data.py` 产出的 `data/train.jsonl`（1:1，跑前算死 50% 多数类基线）。

环境变量（随机标签对照用，PREREG §3）：
  DTSEEK_ANCHORED_TRAIN     指定训练 jsonl（缺省 data/train.jsonl）
  DTSEEK_ANCHORED_SHUF=1    把标签整体 randperm（保持 1:1），span 按新标签重算
  DTSEEK_ANCHORED_SHUF_SEED 打乱用的种子（缺省用 build_dataset 收到的 seed）
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

from dtseek.tasks.plugin import TaskClass, TaskSpec, register

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PREFIX, SEP, SUFFIX = "原文：", "|", "候选："

SPEC = TaskSpec(
    name="anchored_sel",
    label="锚定候选打标",
    classes=(TaskClass(name="背景", label="不支持"),
             TaskClass(name="支持", label="支持")),
    max_steps=1,              # 一个候选一次判定
    max_len=96,               # 实测最长样本 70 字符（data/stats.json），无截断
    emission="single",
    segment_policy="sentence",
    cls_weight_bg=1.0,         # PREREG §3 写死：max_steps=1 时背景类不被稀释，1.3 会引入先验
)


def candidate_span(text: str) -> tuple[int, int]:
    """候选字段在 text 里的 0-based 半开区间（锚点）。"""
    start = text.index(SUFFIX) + len(SUFFIX)
    return start, start + len(text[start:])


class AnchoredCard:
    spec = SPEC

    def build_dataset(self, target_samples: int = 8000, seed: int = 20240927) -> list[dict]:
        path = Path(os.environ.get("DTSEEK_ANCHORED_TRAIN", str(DATA / "train.jsonl")))
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if target_samples != len(rows):
            raise AssertionError(
                f"anchored_sel 数据是**预构建固定文件** {path}（{len(rows)} 条），"
                f"与 --samples {target_samples} 不符 ⇒ 拒绝静默截断/补齐。")
        labels = [int(r["label"]) for r in rows]
        if os.environ.get("DTSEEK_ANCHORED_SHUF") == "1":
            # 随机标签对照：打乱标签（1:1 保持不变），span 按新标签重算
            rng = random.Random(int(os.environ.get("DTSEEK_ANCHORED_SHUF_SEED", seed)))
            rng.shuffle(labels)
        dist = Counter(labels)
        if dist[0] * 2 != len(labels) or dist[1] * 2 != len(labels):
            raise AssertionError(f"标签不是 1:1：{dict(dist)}（多数类基线不再是 50%）")
        print(f"SELFTEST_DATA {path.name} n={len(labels)} 正={dist[1]} 负={dist[0]} "
              f"多数类基线={max(dist.values())/len(labels):.4f} 盲猜=0.5000 "
              f"shuf={os.environ.get('DTSEEK_ANCHORED_SHUF', '0')}", flush=True)
        out = []
        for r, lab in zip(rows, labels):
            s, e = candidate_span(r["text"])
            if e > len(r["text"]) or r["text"][s:e] != r["candidate"]:
                raise AssertionError(f"锚点与候选不一致：{r['id']}")
            out.append({"text": r["text"],
                        "spans": [{"label": 1, "start": s, "end": e}] if lab == 1 else []})
        return out


CARD = register(AnchoredCard())
