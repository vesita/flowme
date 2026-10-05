"""跑 sentiment 单卡基线 + 组合算子，出三项指标与 P1–P3 判定。

    uv run python experiments/compose_ops/run_eval.py

先 fail-closed 校验评测集（cue 必在词典内、否定式整体不得是词典词条），
再按 `preregister.md` 冻结的管线算数；原始逐条结果落 `raw_results.json`。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from make_dataset import build

from dtseek.tasks.builtin.sentiment.dataset import LEXICON_INDEX
from dtseek.tasks.compose import TRIGGERS, decide, filter, flip_rule, pair, to_items
from dtseek.tasks.engine import MultiTaskEngine

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SENTIMENT_CARD = ROOT / "checkpoints/cards/sentiment.pt"
NEGATION_CARD = ROOT / "checkpoints/negation_accept_card.pt"

WINDOW = 8
RULE = flip_rule()

#: 消融 A2：不跑 negation 卡，直接按 TRIGGERS 正则扫原文出标记跨度（长词优先）
_TRIGGER_RE = re.compile("|".join(sorted(TRIGGERS, key=len, reverse=True)))


def regex_markers(text: str) -> list[dict]:
    return [{"label": "否定", "start": m.start(), "end": m.end() - 1, "score": 1.0}
            for m in _TRIGGER_RE.finditer(text)]


def validate(cases: list[dict]) -> None:
    """两条硬约束，任何一条不成立就退出（不许事后调样本）。"""
    bad = []
    for c in cases:
        if c["cue"] not in LEXICON_INDEX:
            bad.append(f"{c['id']}: cue '{c['cue']}' 不在 199 词情绪词典内")
        if c["set"] == "negation":
            # 「否定 + cue」整体不得是词典词条：不存在覆盖 cue 且比 cue 长的词条出现在句中
            for entry in LEXICON_INDEX:
                if len(entry) > len(c["cue"]) and c["cue"] in entry and entry in c["text"]:
                    bad.append(f"{c['id']}: 句中出现词条 '{entry}'（覆盖 cue '{c['cue']}」）")
    if bad:
        raise SystemExit("评测集 fail-closed 校验未过：\n  " + "\n  ".join(bad))


def overlap(a: dict, b: tuple[int, int]) -> bool:
    return a["start"] <= b[1] and b[0] <= a["end"]


def norm(items: list[dict]) -> list[dict]:
    """卡片的 display 名带副标题（“积极/喜悦”），规则与标注都用规范名 → 截首段。"""
    return [{**i, "label": i["label"].split("/", 1)[0]} for i in items]


def run_case(eng: MultiTaskEngine, case: dict) -> dict:
    text = case["text"]
    res = eng.predict(text, tasks=["sentiment", "negation"])
    sent = norm(to_items(res["tasks"].get("sentiment", [])))
    neg = norm(to_items(res["tasks"].get("negation", [])))
    cue = (case["cue_start"], case["cue_end"])

    kept = filter(sent, neg, mode="adjacent", window=WINDOW, clause=True,
                  text=text, require=TRIGGERS, require_side="b")
    comp = pair(sent, kept, RULE, mode="adjacent", window=WINDOW, clause=True, text=text)

    kept_raw = filter(sent, neg, mode="adjacent", window=WINDOW, clause=True, text=text)
    comp_raw = pair(sent, kept_raw, RULE, mode="adjacent", window=WINDOW, clause=True, text=text)

    rgx = regex_markers(text)
    comp_rgx = pair(sent, rgx, RULE, mode="adjacent", window=WINDOW, clause=True, text=text)

    return {
        "id": case["id"], "set": case["set"], "text": text, "cue": case["cue"],
        "label": case["label"], "sanity": case["sanity"],
        "recall": any(overlap(a, cue) for a in sent),
        "recall_strict": any(a["start"] <= cue[0] and cue[1] <= a["end"] for a in sent),
        "base": decide(sent), "comp": decide(comp), "comp_nocheck": decide(comp_raw),
        "comp_regex": decide(comp_rgx),
        "n_sent": len(sent), "n_neg": len(neg), "n_kept": len(kept),
        "sent": sent, "neg": neg, "kept": [k["start"] for k in kept],
        "flipped": [(a["start"], a["end"], b["label"])
                    for a, b in zip(sent, comp) if a["label"] != b["label"]],
    }


def acc(rows: list[dict], key: str) -> float:
    return 100.0 * sum(r[key] == r["label"] for r in rows) / len(rows) if rows else float("nan")


def majority(rows: list[dict]) -> tuple[str, float]:
    if not rows:
        return "-", float("nan")
    lab, n = Counter(r["label"] for r in rows).most_common(1)[0]
    return lab, 100.0 * n / len(rows)


def pct(x: float) -> str:
    return f"{x:.1f}%"


def pol(label: str) -> str:
    """极性口径：只有「积极」算正，其余（含中性）算非正。"""
    return "正" if label == "积极" else "非正"


def pol_acc(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return 100.0 * sum(pol(r[key]) == pol(r["label"]) for r in rows) / len(rows)


def block(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    rec = [r for r in rows if r["recall"]]
    out = {
        "name": name, "n": n,
        "majority": majority(rows),
        "base": acc(rows, "base"),
        "comp": acc(rows, "comp"),
        "comp_nocheck": acc(rows, "comp_nocheck"),
        "comp_regex": acc(rows, "comp_regex"),
        "recall": 100.0 * len(rec) / n if n else float("nan"),
        "recall_strict": 100.0 * sum(r["recall_strict"] for r in rows) / n if n else float("nan"),
        "cond": acc(rec, "comp") if rec else float("nan"),
        "cond_base": acc(rec, "base") if rec else float("nan"),
        "e2e_norecall": acc([r for r in rows if not r["recall"]], "comp"),
        "base_pol": pol_acc(rows, "base"),
        "comp_pol": pol_acc(rows, "comp"),
        "flip_pos": 100.0 * sum(r["comp"] == "积极" for r in rows) / n if n else float("nan"),
        "base_pos": 100.0 * sum(r["base"] == "积极" for r in rows) / n if n else float("nan"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="组合算子评测（默认 ckpt = 预注册那次运行）")
    ap.add_argument("--base", default=str(ROOT / "checkpoints/base_encoder.pt"))
    ap.add_argument("--sentiment", default=str(SENTIMENT_CARD))
    ap.add_argument("--negation", default=str(NEGATION_CARD))
    ap.add_argument("--tag", default="", help="产物文件名后缀（跑 ckpt 敏感性时用）")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    data = build()
    cases = data["cases"]
    validate(cases)

    eng = MultiTaskEngine(base_path=args.base, cards_dir=None)
    eng.attach(args.sentiment)
    eng.attach(args.negation)

    rows = [run_case(eng, c) for c in cases]
    (HERE / f"raw_results{suf}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    by = {k: [r for r in rows if r["set"] == k] for k in ("negation", "control", "stress")}
    by["control_plain"] = [r for r in by["control"] if r["id"].startswith("ctrl_plain")]
    by["control_neg"] = [r for r in by["control"] if r["id"].startswith("ctrl_neg")]
    by["sanity"] = [r for r in rows if r["sanity"]]

    blocks = {k: block(k, v) for k, v in by.items()}
    neg, ctrl = blocks["negation"], blocks["control"]

    # 否定集失败归因（描述性，不参与判据）；「仍判积极」的三类互斥、按优先级归档
    fails = [r for r in by["negation"] if r["comp"] != r["label"]]
    still_pos = [r for r in fails if r["comp"] == "积极"]
    buckets = {"召回失败": [], "无可用否定标记": [], "锚点文本未含情绪词": []}
    for r in still_pos:
        if not r["recall"]:
            buckets["召回失败"].append(r["id"])
        elif not r["kept"]:
            buckets["无可用否定标记"].append(r["id"])
        else:
            buckets["锚点文本未含情绪词"].append(r["id"])
    decomp = {
        "错题总数": len(fails),
        "极性已对、桶错（悲伤↔愤怒）": sum(
            1 for r in fails if pol(r["comp"]) == pol(r["label"])),
        "仍判积极": len(still_pos),
        **{f"  其中{k}": len(v) for k, v in buckets.items()},
        "组合把基线对的改错": sum(
            1 for r in by["negation"] if r["base"] == r["label"] and r["comp"] != r["label"]),
        "基线错、组合改对": sum(
            1 for r in by["negation"] if r["base"] != r["label"] and r["comp"] == r["label"]),
        "仍判积极明细": buckets,
    }

    verdict = {}
    verdict["smoke"] = ("本靶子无空间" if neg["base"] >= 80 else "有空间")
    verdict["P1"] = (neg["comp"] - neg["base"] >= 15.0) and (neg["comp"] >= 80.0)
    verdict["P2"] = neg["recall"] >= 70.0
    verdict["P3"] = (ctrl["comp"] >= ctrl["base"] - 3.0)
    verdict["delta"] = neg["comp"] - neg["base"]
    verdict["ctrl_delta"] = ctrl["comp"] - ctrl["base"]

    out = {"counts": data["counts"], "blocks": blocks, "verdict": verdict,
           "decomp": decomp,
           "cards": {"base": args.base, "sentiment": args.sentiment,
                     "negation": args.negation}}
    (HERE / f"summary{suf}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")

    print(json.dumps({"counts": data["counts"], "verdict": verdict, "decomp": decomp},
                     ensure_ascii=False, indent=1))
    for k, b in blocks.items():
        print(f"\n[{k}] n={b['n']}  多数类={b['majority'][0]} {b['majority'][1]:.1f}%")
        print(f"  基线 E_base={pct(b['base'])}  组合 E={pct(b['comp'])}  Δ={b['comp'] - b['base']:+.1f}pt")
        print(f"  召回 R={pct(b['recall'])}(严格覆盖 cue {pct(b['recall_strict'])})  "
              f"条件成功率 C={pct(b['cond'])}  C_base={pct(b['cond_base'])}  "
              f"未召回时对={pct(b['e2e_norecall'])}")
        print(f"  极性(正/非正) 基线={pct(b['base_pol'])} → 组合={pct(b['comp_pol'])}   "
              f"正向占比 {pct(b['base_pos'])} → {pct(b['flip_pos'])}")
        print(f"  消融A1(无触发校验)={pct(b['comp_nocheck'])}  消融A2(正则标记)={pct(b['comp_regex'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
