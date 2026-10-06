"""价值观卡（value_judge）—— 任务声明 + 数据入口。

只在**本实验进程内**注册（`run_card.py` import 后再委托 `training/train_task_card.py`），
不改 `src/`、不改注册表源文件：注册表是运行时 dict，import 即注册。

数据来自 `build_data.py` 产出的 `data/train.jsonl`（预构建固定文件，类别比 40/30/30，
多数类基线 40%，朴素规则基线 100% —— 见 PREREG §2）。

环境变量（随机标签对照用，PREREG §3）：
  DTSEEK_VALUE_TRAIN     指定训练 jsonl（缺省 data/train.jsonl）
  DTSEEK_VALUE_SHUF=1    把 label 整体 randperm（类别计数不变），span 仍取该样本自己的位置
  DTSEEK_VALUE_SHUF_SEED 打乱用的种子
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

#: 类 id：0=背景（不发射）/ 1=得体 / 2=冒犯（PREREG §1）
CLS_BG, CLS_POL, CLS_OFF = 0, 1, 2
#: 预构建训练集的类别配额（跑前算死）
TRAIN_QUOTA = {CLS_BG: 3240, CLS_POL: 2430, CLS_OFF: 2430}

SPEC = TaskSpec(
    name="value_judge",
    label="价值观判断",
    classes=(TaskClass(name="背景", label="背景/无值判断"),
             TaskClass(name="得体", label="得体/礼貌致谢"),
             TaskClass(name="冒犯", label="冒犯/贬损攻击")),
    max_steps=1,              # 每个样本恰一个值判断（PREREG §3）
    max_len=64,               # 实测最长样本 45 字（data/stats.json），无截断
    emission="single",
    segment_policy="sentence",
    cls_weight_bg=1.0,         # PREREG §3 写死：max_steps=1 时背景类不被稀释，1.3 会引入偏向背景的先验
)


class ValueCard:
    spec = SPEC

    def build_dataset(self, target_samples: int = 8100, seed: int = 20240927) -> list[dict]:
        path = Path(os.environ.get("DTSEEK_VALUE_TRAIN", str(DATA / "train.jsonl")))
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if target_samples != len(rows):
            raise AssertionError(
                f"value_judge 数据是**预构建固定文件** {path}（{len(rows)} 条），"
                f"与 --samples {target_samples} 不符 ⇒ 拒绝静默截断/补齐。")
        labels = [int(r["label"]) for r in rows]
        shuf = os.environ.get("DTSEEK_VALUE_SHUF") == "1"
        if shuf:
            rng = random.Random(int(os.environ.get("DTSEEK_VALUE_SHUF_SEED", seed)))
            rng.shuffle(labels)          # 类别计数不变，只打乱 X–Y 对应
        dist = Counter(labels)
        if not shuf and dict(dist) != TRAIN_QUOTA:
            raise AssertionError(f"训练集类别配额与 PREREG §2 不符：{dict(dist)} != {TRAIN_QUOTA}")
        n = len(labels)
        print(f"SELFTEST_DATA {path.name} n={n} "
              f"背景={dist[CLS_BG]} 得体={dist[CLS_POL]} 冒犯={dist[CLS_OFF]} "
              f"多数类基线={max(dist.values()) / n:.4f} 盲猜=0.3333 "
              f"朴素规则基线(train)=1.0000 "
              f"shuf={'1' if shuf else '0'}", flush=True)
        out = []
        for r, lab in zip(rows, labels):
            s, e = r["span"]
            if not (0 <= s < e <= len(r["text"])):
                raise AssertionError(f"span 越界：{r['id']}")
            if r["entry"] and r["text"][s:e] != r["entry"]:
                raise AssertionError(f"span 没指向条目：{r['id']}")
            out.append({"text": r["text"],
                        "spans": [{"label": lab, "start": s, "end": e}]
                        if lab != CLS_BG else []})
        return out


CARD = register(ValueCard())
