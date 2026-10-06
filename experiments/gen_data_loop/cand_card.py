"""候选有效性卡（`cand_validity`）—— 任务声明 + 数据入口（两臂同一张卡）。

只在**本实验进程内**注册（`run_card.py` import 后再委托 `training/train_task_card.py`），
不改 `src/`、不改注册表源文件：注册表是运行时 dict，import 即注册。

两臂**唯一变量 = 数据来源**，版式、类别体系、max_steps/max_len 全部逐字一致：
  提案臂 P   DTSEEK_CAND_ARM=P  → data/train.jsonl          （8000）
  注入臂 C   DTSEEK_CAND_ARM=C  → data/ctrl_train.jsonl     （8000）

环境变量（随机标签对照，PREREG_PHASE2 §2.1）：
  DTSEEK_CAND_SHUF=1     标签整体 randperm（1:1 计数不变），span 按新标签重算
  DTSEEK_CAND_SHUF_SEED  打乱种子（run_card 传 seed×1000+7）
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
PREFIX, SEP, SUFFIX, CATP = "原文：", "|", "候选：", "类："

ARM_FILES = {"P": DATA / "train.jsonl", "C": DATA / "ctrl_train.jsonl"}

SPEC = TaskSpec(
    name="cand_validity",
    label="候选有效性",
    classes=(TaskClass(name="无效", label="无效"),   # index 0 = 背景，不发射切片
             TaskClass(name="有效", label="有效")),  # index 1 = 发射恰一条
    max_steps=1,            # 一个候选一次判定
    max_len=96,             # PREREG §2 写死；实测最长 text = 96 字符（无截断，见 SELFTEST_TRUNC）
    emission="single",
    segment_policy="sentence",
    cls_weight_bg=1.0,      # max_steps=1 时背景类不被稀释，1.3 会引入先验
)


def candidate_span(text: str) -> tuple[int, int]:
    """候选字段在 text 里的 0-based 半开区间（锚点）：`|候选：<cand>|类：...`"""
    start = text.index(SUFFIX) + len(SUFFIX)
    end = text.index(SEP, start)
    return start, end


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def check_trunc(rows: list[dict], tag: str, tokenizer=None) -> dict:
    """静默截断守卫（PREREG_PHASE2 §4）：文本长度 / span 边界 / 实际编码长度 全部 ≤ max_len。"""
    mx_text = mx_span = 0
    for r in rows:
        L = len(r["text"])
        mx_text = max(mx_text, L)
        if L > SPEC.max_len:
            raise AssertionError(f"[trunc] {tag} {r['id']} text 长 {L} > max_len {SPEC.max_len}")
        for s in r["spans"]:
            mx_span = max(mx_span, s["end"])
            if s["end"] > SPEC.max_len:
                raise AssertionError(f"[trunc] {tag} {r['id']} span.end {s['end']} > max_len")
    mx_enc = -1
    if tokenizer is not None:
        for r in rows:
            enc = tokenizer.encode(r["text"], max_length=SPEC.max_len, padding=True)
            real = int(sum(enc["attention_mask"]))
            if real > SPEC.max_len:
                raise AssertionError(f"[trunc] {tag} {r['id']} 实际编码长度 {real} > max_len")
            mx_enc = max(mx_enc, real)
    info = {"set": tag, "n": len(rows), "max_text_len": mx_text,
            "max_span_end": mx_span, "max_encoded_len": mx_enc,
            "max_len": SPEC.max_len, "overlong": 0}
    print("SELFTEST_TRUNC " + json.dumps(info, ensure_ascii=False), flush=True)
    return info


class CandCard:
    spec = SPEC

    def build_dataset(self, target_samples: int = 8000, seed: int = 20240927) -> list[dict]:
        arm = os.environ.get("DTSEEK_CAND_ARM", "P")
        if arm not in ARM_FILES:
            raise AssertionError(f"DTSEEK_CAND_ARM 必须是 {sorted(ARM_FILES)}，收到 {arm!r}")
        path = ARM_FILES[arm]
        rows = load_jsonl(path)
        if target_samples != len(rows):
            raise AssertionError(
                f"cand_validity 数据是**预构建固定文件** {path}（{len(rows)} 条），"
                f"与 --samples {target_samples} 不符 ⇒ 拒绝静默截断/补齐。")
        labels = [int(r["label"]) for r in rows]
        raw = list(labels)
        if os.environ.get("DTSEEK_CAND_SHUF") == "1":
            # 随机标签对照：整体 randperm（1:1 计数不变），span 按新标签重算
            shuf_seed = int(os.environ.get("DTSEEK_CAND_SHUF_SEED", 0))
            rng = random.Random(shuf_seed)
            rng.shuffle(labels)
            if Counter(labels) != Counter(raw):
                raise AssertionError(f"randperm 改变了标签计数：{dict(Counter(labels))}")
        dist = Counter(labels)
        if dist[0] * 2 != len(labels) or dist[1] * 2 != len(labels):
            raise AssertionError(f"标签不是 1:1：{dict(dist)}（多数类基线不再是 50%）")
        if labels != raw and os.environ.get("DTSEEK_CAND_SHUF") != "1":
            raise AssertionError("标签被意外改动")
        out = []
        for r, lab in zip(rows, labels):
            s, e = candidate_span(r["text"])
            if e <= s or r["text"][s:e] != r["candidate"]:
                raise AssertionError(f"锚点与候选不一致：{r['id']}")
            out.append({"text": r["text"],
                        "spans": [{"label": 1, "start": s, "end": e}] if lab == 1 else []})
        shuf = os.environ.get("DTSEEK_CAND_SHUF", "0")
        info = {"arm": arm, "file": path.name, "n": len(labels),
                "pos": dist[1], "neg": dist[0],
                "majority_baseline": max(dist.values()) / len(labels),
                "blind_guess": 0.5, "shuf": shuf,
                "shuf_seed": os.environ.get("DTSEEK_CAND_SHUF_SEED", None),
                "label_order_changed": labels != raw}
        print("SELFTEST_DATA " + json.dumps(info, ensure_ascii=False), flush=True)
        return out


CARD = register(CandCard())
