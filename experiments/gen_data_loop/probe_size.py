#!/usr/bin/env python3
"""尺寸探针：三张卡的**真值源**在真实语料句子上能出多少候选（纯 CPU、只读、只打印）。

用来定 build_data.py 的池子大小与配额；不写任何文件。
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.builtin.negation.dataset import extract_negation_spans, reject_reason  # noqa: E402
from dtseek.tasks.builtin.pronoun.dataset import extract_all_spans  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import extract_emotion_spans  # noqa: E402
from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402

SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
CJK = re.compile(r"[一-龥]")
BAD = re.compile(r"[�\t｜|]|原文：|候选：")
N = 4000


def cjk(s: str) -> int:
    return len(CJK.findall(s))


def ok(s: str) -> bool:
    if not (14 <= len(s) <= 44) or BAD.search(s):
        return False
    return cjk(s) >= 0.6 * len(s) and not re.search(r"[A-Za-z]{3,}", s)


def sentences(n: int) -> list[str]:
    files = resolve_corpus_files(CORPUS_GLOB)
    random.Random(20240927).shuffle(files)
    out, seen = [], set()
    for p in files:
        with open(p, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                t = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(t):
                    s = s.strip()
                    if s in seen or not ok(s):
                        continue
                    seen.add(s)
                    out.append(s)
                if len(out) >= n:
                    return out
    return out


def main() -> int:
    sents = sentences(N)
    per_card: dict[str, Counter] = {}
    usable: Counter = Counter()
    gold_per_sent: list[int] = []
    covered = 0
    cat_hist: Counter = Counter()
    for s in sents:
        n_gold = 0
        # negation
        if reject_reason(s) is None:
            usable["negation"] += 1
            sp = extract_negation_spans(s)
            if 0 < len(sp) <= 4:
                n_gold += len(sp)
                cat_hist["否定"] += len(sp)
        # sentiment
        lab, sp = extract_emotion_spans(s)
        if lab != -1:
            usable["sentiment"] += 1
            if 0 < len(sp) <= 4:
                n_gold += len(sp)
                cat_hist["sentiment_words"] += len(sp)
        # pronoun
        usable["pronoun"] += 1
        sp = extract_all_spans(s)
        if 0 < len(sp) <= 4:
            n_gold += len(sp)
            cat_hist["pronoun_words"] += len(sp)
        gold_per_sent.append(n_gold)
        covered += int(n_gold > 0)
    n = len(sents)
    print(f"[size] {n} 句 | 有候选的句子 {covered} = {covered/n:.4f}")
    print(f"[size] 卡可用句数：{dict(usable)}")
    print(f"[size] gold 总数 {sum(gold_per_sent)} | 每句均值 {sum(gold_per_sent)/n:.3f}")
    print(f"[size] 每句 gold 数分布 {dict(Counter(gold_per_sent))}")
    print(f"[size] 类别计数 {dict(cat_hist)}")
    # 需要多少句才能凑齐 train 4000 正例
    print(f"[size] 每句正例 ≈ {sum(gold_per_sent)/n:.2f} ⇒ 4000 正例需 ≈ "
          f"{int(4000 / max(1e-9, sum(gold_per_sent)/n))} 句")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
