#!/usr/bin/env python3
"""S2 —— 候选卡在线上核上的端到端指标（`respond` 口径，PREREG §2 步骤 2）。

用法：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
      uv run python experiments/fix_negation/s2_e2e.py [--cards p1,p2] [--limit N]

口径 = `experiments/e2e_d1/w0_e2e.py::neg_metrics`（M_detect/M_cls/M_span，首发射）：
  * 引擎 = 线上核 `checkpoints/base_encoder.pt` + 默认四卡 + 待评卡（只换这一张）；
  * 输入 = negation `eval_S`（`capability_map/cache/negation_ordered_s{42,43}.pkl`，同 `split_of` 公式）；
  * `class_map` 取自被评卡自己的 spec（classes = ['背景','否定'] ⇒ {'背景':0,'否定':1}）。
选卡顺序写死在 PREREG：先 `checkpoints/negation_accept_card.pt`（主候选），不达标才评 `_e12`。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

from common import (  # noqa: E402
    FALLBACK_CARD, HERE, NEG, ONLINE_BASE, PRIMARY_CANDIDATE, ROOT, SEEDS,
    build_engine, neg_rate_summary, neg_vectors, paired_delta, rate, run_respond,
    split_of_e2e,
)


def eval_card(path: str, items_by_seed: dict, class_map_cache: dict) -> dict:
    eng = build_engine(path, base_path=ONLINE_BASE)
    class_map_cache.setdefault(path, {c.name: i for i, c in enumerate(eng.specs[NEG].classes)})
    cm = class_map_cache[path]
    out: dict = {"path": path, "per_seed": {}}
    for S in SEEDS:
        items = items_by_seed[S]
        texts = [it["text"] for it in items]
        recs = run_respond(eng, texts)
        v = neg_vectors(items, recs, cm)
        out["per_seed"][str(S)] = {
            "n": len(items), "summary": neg_rate_summary(v),
            "confusion": v["confusion"], "vectors": v,
            "trigger": sum(1 for r in recs if NEG in r["cards_run"]) / max(1, len(recs)),
            "kind": {"text": sum(1 for r in recs if r["kind"] == "text"),
                     "reject": sum(1 for r in recs if r["kind"] == "reject")},
        }
        s = out["per_seed"][str(S)]["summary"]
        print(f"  [{path} s{S}] n={len(items)} M_detect={s['detect']['rate']:.4f} "
              f"M_cls={s['cls']['rate']:.4f}±{s['cls']['se']:.4f} "
              f"M_span={s['span']['rate']:.4f} conf={v['confusion']}", flush=True)
    del eng
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default=PRIMARY_CANDIDATE, help="逗号分隔的候选卡路径")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None, help="结果文件名后缀（默认按卡名自动）")
    args = ap.parse_args(argv)

    t0 = time.time()
    cards = [FALLBACK_CARD] + [c for c in args.cards.split(",") if c and c != FALLBACK_CARD]
    items_by_seed = {S: (split_of_e2e(NEG, S)[:args.limit] if args.limit else split_of_e2e(NEG, S))
                     for S in SEEDS}

    out: dict = {"prereg": "experiments/fix_negation/PREREG.md",
                 "entry": "dtseek.tasks.dialogue.respond(engine, text, type_='plain')",
                 "base": ONLINE_BASE, "caps": [NEG], "seeds": list(SEEDS),
                 "eval_files": [f"negation_ordered_s{S}.pkl" for S in SEEDS],
                 "n_per_seed": {str(S): len(items_by_seed[S]) for S in SEEDS},
                 "cards": {}}
    print(f"[entry] {out['entry']}  base={ONLINE_BASE}", flush=True)
    cm_cache: dict = {}
    for p in cards:
        print(f"[card] {p}", flush=True)
        out["cards"][p] = eval_card(p, items_by_seed, cm_cache)

    # 配对 Δ：候选 − 现默认（同批输入、同 seed）
    out["paired_vs_current"] = {}
    for p in cards[1:]:
        d = {}
        for m in ("detect", "cls", "span"):
            d[m] = {}
            for S in SEEDS:
                a = out["cards"][FALLBACK_CARD]["per_seed"][str(S)]["vectors"][m]   # 现默认（基线）
                b = out["cards"][p]["per_seed"][str(S)]["vectors"][m]               # 候选
                d[m][str(S)] = paired_delta(a, b)   # delta = mean(b − a) = 候选 − 现默认
        out["paired_vs_current"][p] = d
    print("\n=== 候选 vs 现默认（Δ = 候选 − 现默认，配对、同批输入）===")
    for p, d in out["paired_vs_current"].items():
        for m, per in d.items():
            print(f"  {p} M_{m:6s} " + "  ".join(
                f"s{S}: {per[str(S)]['delta']:+.4f}±{per[str(S)]['se']:.4f}"
                f"{'*' if per[str(S)]['significant'] else ''}" for S in SEEDS))
    print(f"[time] {time.time() - t0:.0f}s")

    tag = args.tag or ("_".join(Path(c).stem for c in cards))[:120]
    res = HERE / "results" / f"s2_e2e__{tag}.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}\nS2_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
