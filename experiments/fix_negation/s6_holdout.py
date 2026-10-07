#!/usr/bin/env python3
"""S6 —— **披露项（不是门禁）**：把卡放到**全新生成、与训练集零重叠**的否定数据上再看一次。

动机（实测事实，见 `PREREG_TRAIN_CLARIFY2.md` §0）：既有 benchmark `eval_S` 的文本
**100% 落在各卡训练数据里**（in-sample 口径，旧卡同样如此）⇒ F1 的 M_cls 不能读作泛化指标。
本脚本用**另一个 seed** 现场生成一批样本，先验证与训练集/eval_S **零文本重叠**，再跑同口径端到端指标。

用法：PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s6_holdout.py [--cards a.pt,b.pt]
输出只进 results/s6_holdout.json；**不参与 F0–F5 任何判定**。
"""
from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True

from common import (  # noqa: E402
    FALLBACK_CARD, HERE, NEG, SEEDS, build_engine, lookup, neg_rate_summary,
    neg_vectors, run_respond, split_of_e2e,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.builtin.negation.dataset import build_negation_dataset  # noqa: E402
from dtseek.tasks.plugin import DEFAULT_SEED  # noqa: E402

HOLDOUT_SEED = 90210
HOLDOUT_N = 2000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=FALLBACK_CARD)
    args = ap.parse_args(argv)

    fresh = build_negation_dataset(target_samples=HOLDOUT_N, seed=HOLDOUT_SEED)
    train_txt = {x["text"] for x in build_negation_dataset(target_samples=8000, seed=DEFAULT_SEED)}
    eval_txt = set()
    for S in SEEDS:
        eval_txt |= {it["text"] for it in split_of_e2e(NEG, S)}
    fresh_txt = {x["text"] for x in fresh}
    overlap = {"n_fresh": len(fresh), "vs_train8000": len(fresh_txt & train_txt),
               "vs_evalS": len(fresh_txt & eval_txt)}
    print(f"[holdout] seed={HOLDOUT_SEED} n={len(fresh)} 与 train8000 重叠={overlap['vs_train8000']} "
          f"与 eval_S 重叠={overlap['vs_evalS']}", flush=True)

    # 严格未见子集：既不在 train8000、也不在 eval_S 里
    unseen = [x for x in fresh if x["text"] not in train_txt and x["text"] not in eval_txt]
    print(f"[holdout] 未见子集 n = {len(unseen)}（overlap 已单列）", flush=True)

    out: dict = {"prereg": "experiments/fix_negation/PREREG.md",
                 "note": "披露项，不参与 F0–F5 判定；换 seed 仍有大量文本重叠（见 overlap）",
                 "holdout_seed": HOLDOUT_SEED, "n_fresh": len(fresh), "overlap": overlap,
                 "n_unseen": len(unseen), "floor_note": "地板口径见 results/s4_floor.json（fit=train_S, eval=eval_S）",
                 "cards": {}}
    texts = [x["text"] for x in unseen] if len(unseen) >= 150 else []
    if not texts:
        print("[holdout] 未见子集过小，不做指标（仅报 overlap）")
    for p in ([c for c in args.cards.split(",") if c] if texts else []):
        eng = build_engine(p)
        cm = {c.name: i for i, c in enumerate(eng.specs[NEG].classes)}
        recs = run_respond(eng, texts)
        v = neg_vectors(unseen, recs, cm)
        out["cards"][p] = {"summary": neg_rate_summary(v), "confusion": v["confusion"]}
        s = out["cards"][p]["summary"]
        print(f"  [{p}] M_detect={s['detect']['rate']:.4f} M_cls={s['cls']['rate']:.4f} "
              f"M_span={s['span']['rate']:.4f} conf={v['confusion']}", flush=True)
        del eng

    res = HERE / "results" / "s6_holdout.json"
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}\nS6_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
