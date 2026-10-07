#!/usr/bin/env python3
"""D1 —— 候选卡在线上核上的**错法诊断**（只为定重训配方，不进判据）。

打印 `M_cls` 分母里的错例（无发射 / 类别错），以及真值切片词，
看错在「漏检」还是「类分错」、是否集中在特定形态。
用法：CUDA_VISIBLE_DEVICES="" uv run python experiments/fix_negation/d1_errors.py [--card PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

from common import (  # noqa: E402
    NEG, PRIMARY_CANDIDATE, build_engine, run_respond, split_of_e2e, truth_first,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=PRIMARY_CANDIDATE)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args(argv)

    eng = build_engine(args.card)
    class_map = {c.name: i for i, c in enumerate(eng.specs[NEG].classes)}
    for S in (42, 43):
        items = split_of_e2e(NEG, S)[: args.limit]
        recs = run_respond(eng, [it["text"] for it in items])
        miss, wrong = [], []
        for it, rec in zip(items, recs):
            t = truth_first(it)
            ev = [e for e in rec["evidence"] if e["card"] == NEG]
            if t["label"] <= 0:
                if ev:
                    wrong.append(("FP", it["text"], "", rec))
                continue
            if not ev:
                miss.append(("FN", it["text"], t["sub"], rec))
            elif class_map.get(ev[0]["class_name"], -1) != t["label"]:
                wrong.append(("CLS", it["text"], t["sub"], rec))
        print(f"\n=== s{S} | 漏检 FN {len(miss)} | 类别错/误报 {len(wrong)} ===")
        for tag, text, sub, _rec in miss[: args.show]:
            print(f"  [{tag}] 真值切片={sub!r} | {text.strip()[:70]}")
        for tag, text, sub, _rec in wrong[: 8]:
            print(f"  [{tag}] 真值切片={sub!r} | {text.strip()[:70]}")
        # 按真值切片词看漏检分布
        from collections import Counter
        miss_words = Counter()
        idx = 0
        for it, rec in zip(items, recs):
            t = truth_first(it)
            if t["label"] > 0 and not [e for e in rec["evidence"] if e["card"] == NEG]:
                miss_words[t["sub"]] += 1
        print(f"  漏检按真值切片词 Top: {miss_words.most_common(10)}")
        print(f"  样本按真值切片词 Top: "
              f"{Counter(truth_first(it)['sub'] for it in items if truth_first(it)['label'] > 0).most_common(10)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
