#!/usr/bin/env python3
"""数据独立复核 + 对抗集人工抽检辅助（不复用 build_data 的断言，独立重算）。

输出：
  1) 三个集合的规模 / 正负比 / 多数类基线（重算）
  2) 两两不相交：整条 text / 上下文 / 候选 三口径的重叠数（重算）
  3) 标签与锚点一致性：正例 span 是否真指向候选；负例是否真的不是逐字片段
  4) 对抗集 reorder：把候选还原成「上下文里某个连续片段的同词表重排」（逐条验证）
  5) 抽 20 条 reorder 打印出来供人工判断标注噪声（PREREG §1）
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]


def find_source_frag(ctx: str, cand: str) -> str | None:
    """在 ctx 里找与 cand 同词表的连续片段（reorder 的构造源）。"""
    n = len(cand)
    target = sorted(cand)
    for i in range(len(ctx) - n + 1):
        seg = ctx[i:i + n]
        if sorted(seg) == target:
            return seg
    return None


def main() -> None:
    splits = {k: load(k) for k in ("train", "test", "adversarial")}
    print("== 1) 规模 / 正负比 / 基线 ==")
    for k, rows in splits.items():
        d = Counter(r["label"] for r in rows)
        maj = max(d.values()) / len(rows)
        lex = sum(1 for r in rows if (r["candidate"] in r["context"]) == bool(r["label"])) / len(rows)
        print(f"  {k:13s} n={len(rows):5d} 正{d[1]} 负{d[0]} 多数类={maj:.4f} "
              f"盲猜=0.5000 表层子串规则={lex:.4f} 规则={dict(Counter(r['rule'] for r in rows))}")

    print("== 2) 两两不相交（独立重算）==")
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = splits[names[i]], splits[names[j]]
            ta = {r["text"] for r in a}; tb = {r["text"] for r in b}
            ca = {r["context"] for r in a}; cb = {r["context"] for r in b}
            ka = {r["candidate"] for r in a}; kb = {r["candidate"] for r in b}
            print(f"  {names[i]}∩{names[j]}: text={len(ta & tb)} "
                  f"context={len(ca & cb)} candidate={len(ka & kb)}")

    print("== 3) 标签 / 锚点一致性 ==")
    bad_span = bad_neg = bad_pos = 0
    for k, rows in splits.items():
        for r in rows:
            if r["label"]:
                s = r["spans"][0]
                if r["text"][s["start"]:s["end"]] != r["candidate"]:
                    bad_span += 1
                if k in ("train", "test") and r["rule"] == "verbatim" \
                        and r["candidate"] not in r["context"]:
                    bad_pos += 1
            else:
                if r["candidate"] in r["context"]:
                    bad_neg += 1
    print(f"  span 未指向候选: {bad_span} | train/test 逐字正例不是片段: {bad_pos} | "
          f"负例却是逐字片段: {bad_neg}")

    print("== 4) 对抗集 reorder 同词表还原 ==")
    adv = splits["adversarial"]
    ro = [r for r in adv if r["rule"] == "reorder"]
    found = sum(1 for r in ro if find_source_frag(r["context"], r["candidate"]) is not None)
    print(f"  reorder {len(ro)} 条，能在上下文里找到同词表连续片段的: {found}")
    pa = [r for r in adv if r["rule"] == "paraphrase"]
    inctx = sum(1 for r in pa if r["candidate"] in r["context"])
    print(f"  paraphrase {len(pa)} 条，候选逐字出现在上下文里的（应为 0）: {inctx}")

    print("== 5) 人工抽检 20 条 reorder（判断标注噪声）==")
    rng = random.Random(20240927)
    for r in rng.sample(ro, 20):
        frag = find_source_frag(r["context"], r["candidate"])
        print(f"  ctx={r['context']}")
        print(f"    frag={frag}  cand={r['candidate']}")


if __name__ == "__main__":
    main()
