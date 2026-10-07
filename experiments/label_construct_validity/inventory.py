#!/usr/bin/env python3
"""P5-Q0：候选标签源逐源盘点 —— 定义方式 + 免费规则 acc（rule）+ 可用性判定。

只读：不改任何既有文件；产物只落 experiments/label_construct_validity/results/。

口径（PREREG §2/§3 写死）：
  - `rule` = 用**该标签自身的定义规则**（= 数据集自己的 span/label 提取器）去预测该标签；
  - 评测单位 = **行级 exact match**（span 序列与类别逐位相同才算对）；
  - 可用性：`rule < 1.0 − 2×SE`（SE = sqrt(rule(1-rule)/n)，n = 行数）⇒ 才算「可用」。
"""
from __future__ import annotations

import json
import math
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

import numpy as np  # noqa: E402

from dtseek.tasks.builtin.idiom.dataset import extract_idiom_spans  # noqa: E402
from dtseek.tasks.builtin.negation.dataset import extract_negation_spans  # noqa: E402
from dtseek.tasks.builtin.ownership.dataset import extract_speaker_spans  # noqa: E402
from dtseek.tasks.builtin.pronoun.dataset import extract_all_spans  # noqa: E402
from dtseek.tasks.builtin.relation.dataset import (  # noqa: E402
    PAIR_TABLES,
    find_unannotated_pair,
    lexicon_words_in,
    locate_pair,
)
from dtseek.tasks.builtin.sentiment.dataset import extract_emotion_spans  # noqa: E402
from dtseek.tasks.builtin.person import dataset as person_ds  # noqa: E402

EMO_CACHE = ROOT / "experiments/core_keep/cache/sentiment_32000.pkl"
SKEL_DIR = ROOT / "experiments/two_channel_head/data"
CORE = ROOT / "experiments/core_probe"


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def norm_spans(sp) -> list:
    """把 spans 规范成 [(start,end,label), ...] 按 start 排序（丢弃 word/pair_key 等）。"""
    out = []
    for s in sp or []:
        if isinstance(s, dict):
            out.append((int(s["start"]), int(s["end"]), int(s.get("label", -1))))
        else:
            out.append(tuple(int(x) for x in s))
    return sorted(out)


def row_exact(gold_sp, pred_sp) -> bool:
    return norm_spans(gold_sp) == norm_spans(pred_sp)


def acc_block(name: str, n: int, good: int, dist: Counter, desc: str, ncls: int) -> dict:
    a = good / max(n, 1)
    s = se(a, n)
    return {
        "source": name,
        "n": int(n),
        "n_classes": int(ncls),
        "train_or_src_dist": {str(k): int(v) for k, v in sorted(dist.items())},
        "rule_acc": round(a, 6),
        "rule_se": round(s, 6),
        "threshold_1m2se": round(1.0 - 2 * s, 6),
        "usable": bool(a < 1.0 - 2 * s),
        "rule_desc": desc,
    }


# ── 1. 情绪 emo ────────────────────────────────────────────────────────────
def do_emo() -> dict:
    rows = pickle.load(open(EMO_CACHE, "rb"))
    good = 0
    dist = Counter(int(r["label"]) for r in rows)
    abst = 0
    for r in rows:
        dom, _ = extract_emotion_spans(r["text"])
        pred = dom if dom in (0, 1, 2, 3) else 0
        abst += int(dom == -1)
        good += int(pred == r["label"])
    res = acc_block(
        "emo / sentiment_32000.pkl（全量 cache）", len(rows), good, dist,
        f"extract_emotion_spans 词表（EMOTION_KEYWORDS ~200 词），弃权判 0，弃权 n={abst}", 4)

    # core_probe 的 test split（同 rng42 抽样口径）上再算一次 —— Q0 的主口径
    per: dict = {c: [] for c in range(4)}
    for r in rows:
        per[r["label"]].append(r)
    rng = random.Random(42)
    te = []
    for c in range(4):
        idx = list(range(len(per[c])))
        rng.shuffle(idx)
        sel = [per[c][i] for i in idx[:2000]]
        te += sel[1000:]
    g2 = d2 = 0
    for r in te:
        dom, _ = extract_emotion_spans(r["text"])
        pred = dom if dom in (0, 1, 2, 3) else 0
        d2 += int(dom == -1)
        g2 += int(pred == r["label"])
    res_test = acc_block(
        "emo / core_probe test split（n=4000，rng42 同口径）", len(te), g2,
        Counter(int(r["label"]) for r in te),
        f"同上；弃权判 0，弃权 n={d2}", 4)
    return {"cache": res, "core_probe_test": res_test}


# ── 2. 逻辑 logic ─────────────────────────────────────────────────────────
def do_logic() -> dict:
    keys = list(json.loads(open(SKEL_DIR / "train.jsonl").readline()).keys())
    return {
        "stored_label_field": keys,
        "has_stored_label": any("logic" in k.lower() for k in keys),
        "note": ("logic 标签**不落盘**：two_channel_head/data/train.jsonl 无 logic 字段；"
                 "core_probe 在加载时用 LOGIC_WORDS(19 词) 现算（run_probe.py:139）。"
                 "⇒ 标签定义 = 词表匹配，rule 按定义 = 1.0000（构造性满分）。"),
        "rule_acc": 1.0,
        "rule_se": 0.0,
        "usable": False,
        "n": 0,
    }


# ── 3. 结构 skel ──────────────────────────────────────────────────────────
def do_skel() -> dict:
    st = json.loads((SKEL_DIR / "stats.json").read_text())
    n_tr = sum(1 for _ in open(SKEL_DIR / "train.jsonl"))
    n_te = sum(1 for _ in open(SKEL_DIR / "test.jsonl"))
    a = float(st["max_naive_train"])
    s = se(a, n_tr)
    dist = Counter(json.loads(l)["skel_id"] for l in open(SKEL_DIR / "train.jsonl"))
    return {
        "source": "skel / two_channel_head train.jsonl::skel_id",
        "n": n_tr, "n_test": n_te, "n_classes": int(st["n_skeletons"]),
        "train_dist_top5": {str(k): int(v) for k, v in dist.most_common(5)},
        "n_distinct_train": len(dist),
        "rule_acc": round(a, 6), "rule_se": round(s, 6),
        "threshold_1m2se": round(1.0 - 2 * s, 6),
        "usable": bool(a < 1.0 - 2 * s),
        "rule_desc": ("two_channel_head 自带免费规则套件 max_naive_train（stats.json 引用实测）；"
                      "标签本身 = 模板生成器按 skel 模板产句后回填 skel_id（构造性）"),
        "label_origin": "构造性（模板生成器）",
    }


# ── 4. 身份-回忆 idr ──────────────────────────────────────────────────────
def do_idr() -> dict:
    rows = [r for r in pickle.load(open(ROOT / "experiments/core_keep/cache/person_6000.pkl", "rb"))
            if r["spans"]]
    tot = good = 0
    for r in rows:
        seen_lab: set = set()
        seen_word: set = set()
        for s in r["spans"]:
            gold = 1 if s["label"] in seen_lab else 0
            pred = 1 if s["word"] in seen_word else 0
            tot += 1
            good += int(gold == pred)
            seen_lab.add(s["label"])
            seen_word.add(s["word"])
    a = good / tot
    s = se(a, tot)
    return {"source": "idr / person_6000.pkl 提及级 is_repeat",
            "n": tot, "n_classes": 2, "rule_acc": round(a, 6), "rule_se": round(s, 6),
            "threshold_1m2se": round(1.0 - 2 * s, 6), "usable": bool(a < 1.0 - 2 * s),
            "rule_desc": "字面重复规则（surface 词在本文档前文出现过 ⇒ repeat）",
            "label_origin": "构造性（生成器 + 复核解析器），但规则只复现 0.79"}


# ── 5. person 全 id（首现顺序）────────────────────────────────────────────
def do_person_id() -> dict:
    rows = pickle.load(open(ROOT / "experiments/core_keep/cache/person_6000.pkl", "rb"))
    good = n = dropped = 0
    dist = Counter()
    for r in rows:
        try:
            scan = person_ds._scan_person_mentions(r["text"])
        except Exception:
            dropped += 1
            continue
        # 规则：词表扫描出提及 → 按首现顺序编号（= 数据集 docstring 写死的定义）
        spans = []
        surf: dict = {}
        for m in scan:
            w = m["word"] if "word" in m else r["text"][m["start"]:m["end"]]
            if w not in surf:
                surf[w] = len(surf) + 1
            spans.append({"start": m["start"], "end": m["end"], "label": surf[w]})
        n += 1
        good += int(row_exact(r["spans"], spans))
        for s in r["spans"]:
            dist[int(s["label"])] += 1
    a = good / max(n, 1)
    s = se(a, n)
    return {"source": "person / person_6000.pkl 人物 id（首现顺序）",
            "n": n, "dropped": dropped, "n_classes": len(dist),
            "mention_label_dist_top6": {str(k): int(v) for k, v in dist.most_common(6)},
            "rule_acc": round(a, 6), "rule_se": round(s, 6),
            "threshold_1m2se": round(1.0 - 2 * s, 6), "usable": bool(a < 1.0 - 2 * s),
            "rule_desc": "_scan_person_mentions 词表扫描 + 按首现顺序编号（docstring 定义）；行级 exact match",
            "label_origin": "构造性（生成器 + 独立复核解析器）"}


# ── 6. 其余 span 卡（同构：用各自的提取器当规则）─────────────────────────
def do_span_task(name: str, path: str, extractor, dist_field=None, extra=None) -> dict:
    rows = pickle.load(open(ROOT / path, "rb"))
    good = n = 0
    dist = Counter()
    errs = 0
    for r in rows:
        try:
            pred = extractor(r["text"])
        except Exception:
            errs += 1
            continue
        if pred is None:
            pred = []
        n += 1
        good += int(row_exact(r["spans"], pred))
        if dist_field and dist_field in r:
            dist[str(r[dist_field])] += 1
    a = good / max(n, 1)
    s = se(a, n)
    out = {"source": name, "n": n, "errors": errs,
           "n_rows_total": len(rows),
           "rule_acc": round(a, 6), "rule_se": round(s, 6),
           "threshold_1m2se": round(1.0 - 2 * s, 6), "usable": bool(a < 1.0 - 2 * s),
           "rule_desc": extra or f"{extractor.__module__.split('.')[-2]}.{extractor.__name__}（该标签的定义本身）；行级 exact match",
           "label_origin": "构造性（词表/规则生成器）"}
    if dist:
        out["dist"] = dict(dist.most_common(8))
    return out


def do_relation() -> dict:
    rows = pickle.load(open(ROOT / "experiments/core_keep/cache/relation_6000.pkl", "rb"))
    syn = set(PAIR_TABLES[1])
    ant = set(PAIR_TABLES[2])
    good = n = 0
    dist = Counter()
    for r in rows:
        text = r["text"]
        gold_cat = int(r.get("category") or 0)     # 背景行没有 category 键 ⇒ 0
        pair = find_unannotated_pair(text)
        if pair is None:
            pred_cat, pred_sp = 0, []
        else:
            key = (pair[0], pair[1])
            pred_cat = 1 if key in syn else (2 if key in ant else -1)
            sp = locate_pair(text, pair[0], pair[1])
            pred_sp = [{"start": s["start"], "end": s["end"], "label": pred_cat}
                       for s in (sp or [])]
        n += 1
        if gold_cat == 0:                          # 背景行：规则只要别误报出完整词对
            good += int(pred_cat == 0)
        else:                                      # 正例：类别 + span 序列都对
            good += int(pred_cat == gold_cat and row_exact(r["spans"], pred_sp))
        dist[str(gold_cat)] += 1
    a = good / max(n, 1)
    s = se(a, n)
    return {"source": "relation / relation_6000.pkl（近/反义词对类别）",
            "n": n, "n_classes": len(dist), "dist": dict(dist),
            "rule_acc": round(a, 6), "rule_se": round(s, 6),
            "threshold_1m2se": round(1.0 - 2 * s, 6), "usable": bool(a < 1.0 - 2 * s),
            "rule_desc": "PAIR_TABLES 词表 + find_unannotated_pair/locate_pair（标签定义本身）；行级 exact match",
            "label_origin": "构造性（词表对 + 关系中立帧模板）"}


# ── 7. value_card 手写金标准（仓内唯一 hand 标注，作旁证）─────────────────
def do_value_gold() -> dict:
    sys.path.insert(0, str(ROOT / "experiments/value_card"))
    rows = [json.loads(l) for l in open(ROOT / "experiments/value_card/data/gold.jsonl")]
    # 规则 = value_card 的 OFF/POL/NEU 词表（build_data.py 的三张表）
    import build_data as bd  # type: ignore
    off, pol = set(bd.OFF), set(bd.POL)
    good = n = 0
    dist = Counter()
    for r in rows:
        text, lab = r["text"], int(r["label"])
        span = r["span"]
        word = text[span[0]:span[1]] if span and len(span) == 2 else ""
        if lab == 0:
            pred = 0 if not (any(w in text for w in off) or any(w in text for w in pol)) else 1
        else:
            table = off if lab == 2 else pol
            pred = lab if word in table else 0
        n += 1
        good += int(pred == lab)
        dist[str(lab)] += 1
    a = good / max(n, 1)
    s = se(a, n)
    return {"source": "value_card/gold.jsonl（手写金标准，rule=hand，n=120，40/40/40）",
            "n": n, "n_classes": len(dist), "dist": dict(dist),
            "rule_acc": round(a, 6), "rule_se": round(s, 6),
            "threshold_1m2se": round(1.0 - 2 * s, 6), "usable": bool(a < 1.0 - 2 * s),
            "rule_desc": "value_card OFF/POL/NEU 三张词表去预测手写标签（类别级）",
            "label_origin": "外部代理手写（V3 外部代理真值）；**非情绪/非逻辑**构念（得体/冒犯/背景）"}


def main() -> int:
    res: dict = {}
    res["emo"] = do_emo()
    res["logic"] = do_logic()
    res["skel"] = do_skel()
    res["idr"] = do_idr()
    res["person_id"] = do_person_id()
    res["pronoun"] = do_span_task(
        "pronoun / pronoun_6000.pkl", "experiments/core_keep/cache/pronoun_6000.pkl",
        extract_all_spans)
    res["negation"] = do_span_task(
        "negation / negation_6000.pkl", "experiments/core_keep/cache/negation_6000.pkl",
        extract_negation_spans)
    res["ownership"] = do_span_task(
        "ownership / ownership_6000.pkl", "experiments/core_generalize/cache/ownership_6000.pkl",
        extract_speaker_spans)
    res["idiom"] = do_span_task(
        "idiom / idiom_6000.pkl", "experiments/core_generalize/cache/idiom_6000.pkl",
        extract_idiom_spans)
    res["relation"] = do_relation()
    res["value_gold"] = do_value_gold()

    (OUT / "inventory.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    print("\n| 源 | 定义方式 | n | 类别数 | rule | SE | 门槛 1-2SE | 可用? |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in res.items():
        if k == "logic":
            print("| logic | 加载时词表现算（标签不落盘） | — | 2 | **1.0000**（按定义） | 0 | — | 否 |")
            continue
        if k == "emo":
            for sub in ("cache", "core_probe_test"):
                x = v[sub]
                print(f"| {x['source']} | 词表生成器 | {x['n']} | {x['n_classes']} | "
                      f"**{x['rule_acc']:.4f}** | {x['rule_se']:.5f} | {x['threshold_1m2se']:.4f} | "
                      f"{'是' if x['usable'] else '否'} |")
            continue
        print(f"| {v['source']} | {'构造性' if '构造性' in str(v.get('label_origin','')) else '外部手写'} | "
              f"{v['n']} | {v.get('n_classes','—')} | **{v['rule_acc']:.4f}** | {v['rule_se']:.5f} | "
              f"{v['threshold_1m2se']:.4f} | {'是' if v['usable'] else '否'} |")
    print(f"\n写入 {OUT / 'inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
