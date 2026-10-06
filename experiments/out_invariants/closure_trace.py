"""闭合性 + 可溯源 的代码检查 + 一次真实组合调用（纯推理，不训练）。

    uv run python experiments/out_invariants/closure_trace.py 2>&1 | tee experiments/out_invariants/logs/closure_trace.log

产物：experiments/out_invariants/closure.json
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.compose import decide, filter, flip_rule, pair, to_items  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402

CARDS = ["pronoun", "sentiment", "relation", "person", "negation"]
TEXT = "这个方案不难，我很喜欢，他没有反对。"


def keys_of(objs: list[dict]) -> dict:
    """并集键 + 每个键的出现次数 + 一次示例类型。"""
    out: dict[str, dict] = {}
    for o in objs:
        for k, v in o.items():
            e = out.setdefault(k, {"n": 0, "type": type(v).__name__})
            e["n"] += 1
    return out


def attempt(fn, *a, **kw) -> dict:
    try:
        r = fn(*a, **kw)
        return {"ok": True, "result": r if not isinstance(r, list) else r[:3],
                "len": len(r) if hasattr(r, "__len__") else None}
    except Exception as exc:  # noqa: BLE001  这里就是要捕获失败类型
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback_last": traceback.format_exc().splitlines()[-1]}


def main() -> int:
    import os
    os.chdir(ROOT)
    engine = MultiTaskEngine(base_path="checkpoints/base_encoder.pt", cards_dir="checkpoints/cards")
    engine.attach("checkpoints/negation_accept_card_e12.pt")

    res = engine.predict(TEXT, tasks=CARDS)
    anchors = {k: v for k, v in res["tasks"].items() if v}
    all_anchors = [a for v in res["tasks"].values() for a in v]

    # ---- 1. 结构对比 ------------------------------------------------------
    structure = {
        "predict_返回顶层键": sorted(res.keys()),
        "锚点键（引擎产物）": keys_of(all_anchors),
        "各卡实际产出条数": {k: len(v) for k, v in res["tasks"].items()},
    }

    # ---- 2. 区间语义 ------------------------------------------------------
    interval = {}
    for k, v in anchors.items():
        for a in v:
            closed = TEXT[a["s0"]:a["e0"] + 1]
            half = TEXT[a["s0"]:a["e0"]]
            interval.setdefault(k, []).append(
                {"s0": a["s0"], "e0": a["e0"], "闭区间切片": closed, "半开区间切片": half})
    # 数据集侧（监督真值）的口径
    spec = resolve_tasks(["sentiment"])["sentiment"]
    sample = next(s for s in spec.build_dataset(400) if s.get("spans"))
    sp = sample["spans"][0]
    dataset_semantics = {
        "dataset_span_示例": sp,
        "半开解释 start:end": sample["text"][sp["start"]:sp["end"]],
        "闭解释 start:end+1": sample["text"][sp["start"]:sp["end"] + 1],
        "协议声明（plugin.py:276）": "0-based 半开区间",
    }

    # ---- 3. 产物类型转换与回喂 -------------------------------------------
    sen = res["tasks"]["sentiment"]
    neg = res["tasks"]["negation"]
    items_sen = to_items(sen)
    items_neg = to_items(neg)

    combined = pair(items_sen, items_neg, flip_rule(), mode="adjacent", window=8,
                    clause=True, text=TEXT)
    filtered = filter(items_sen, items_neg, mode="overlap")

    closure = {
        "引擎锚点键": sorted({k for a in all_anchors for k in a}),
        "to_items 产物键": sorted({k for a in items_sen + items_neg for k in a}),
        "filter 产物键": sorted({k for a in filtered for k in a}),
        "pair 产物键": sorted({k for a in combined for k in a}),
        "回喂 to_items（算子产物→适配器）": attempt(to_items, combined),
        "回喂 filter（算子产物→算子）": attempt(filter, combined, items_neg),
        "回喂 pair（算子产物→算子）": attempt(pair, combined, items_neg, flip_rule()),
        "回喂引擎 render（算子产物→呈现）": attempt(MultiTaskEngine.render, TEXT, combined),
        "引擎锚点直喂 filter（引擎产物→算子）": attempt(filter, sen, neg),
        "引擎锚点直喂 pair（引擎产物→算子）": attempt(pair, sen, neg, flip_rule()),
        "decide(算子产物)": attempt(decide, combined),
        "组合真实调用": {
            "text": TEXT,
            "sentiment_锚点": sen,
            "negation_锚点": neg,
            "to_items(sentiment)": items_sen,
            "pair(sentiment, negation, flip)": combined,
            "filter(sentiment, negation, overlap)": filtered,
            "句级结论": decide(combined),
        },
    }

    # 第二个真实组合调用：分句内否定 ⇒ pair 的 flip 真正生效
    TEXT2 = "我不喜欢这个方案。"
    res2 = engine.predict(TEXT2, tasks=["sentiment", "negation"])
    sen2, neg2 = res2["tasks"]["sentiment"], res2["tasks"]["negation"]
    combo2 = {
        "text": TEXT2,
        "sentiment_锚点": sen2, "negation_锚点": neg2,
        "to_items(sentiment)": to_items(sen2),
        "pair(带 clause=True)": pair(to_items(sen2), to_items(neg2), flip_rule(),
                                     mode="adjacent", window=8, clause=True, text=TEXT2),
        "pair(不带 clause)": pair(to_items(sen2), to_items(neg2), flip_rule(),
                                  mode="adjacent", window=8, clause=False, text=TEXT2),
        "filter(overlap)": filter(to_items(sen2), to_items(neg2), mode="overlap"),
        "decide": decide(pair(to_items(sen2), to_items(neg2), flip_rule(),
                              mode="adjacent", window=8, clause=True, text=TEXT2)),
    }

    # 第三个真实调用：flip 规则的默认 apply_to 与卡的真实 display 名是否同源
    TEXT3 = "我一点也不欣赏他的做法。"
    res3 = engine.predict(TEXT3, tasks=["sentiment", "negation"])
    s3, n3 = res3["tasks"]["sentiment"], res3["tasks"]["negation"]
    norm = lambda xs: [{**i, "label": i["label"].split("/", 1)[0]} for i in xs]  # noqa: E731
    combo3 = {
        "text": TEXT3,
        "sentiment_锚点": s3, "negation_锚点": n3,
        "card_display_label": [a["category"] for a in s3],
        "flip_rule()_默认 apply_to": list(flip_rule()["apply_to"]),
        "pair_默认规则（引擎原始标签）": pair(to_items(s3), to_items(n3), flip_rule(),
                                             mode="adjacent", window=8, clause=True, text=TEXT3),
        "pair_显式 apply_to=积极/喜悦": pair(to_items(s3), to_items(n3),
                                             flip_rule(apply_to=("积极/喜悦",)),
                                             mode="adjacent", window=8, clause=True, text=TEXT3),
        "pair_调用方先 norm() 再喂": pair(norm(to_items(s3)), norm(to_items(n3)), flip_rule(),
                                          mode="adjacent", window=8, clause=True, text=TEXT3),
    }
    closure["组合真实调用3_标签不同源"] = combo3

    closure["组合真实调用2_同分句否定"] = combo2

    # ---- 4. 可溯源 --------------------------------------------------------
    have = set(res.keys()) | {f"tasks.*.{k}" for a in all_anchors for k in a}
    trace = {
        "predict 返回顶层键": sorted(res.keys()),
        "锚点键": sorted({k for a in all_anchors for k in a}),
        "engine 上存在但**没有**进返回值的字段": {
            "module_id": "锚点里没有；只在 result['tasks'] 的字典键上（单条锚点脱离字典即丢失卡名）",
            "输入": "result['text'] 有整段原文；**没有**该锚点来自哪个 segment、segment 的原文与偏移",
            "参数": [
                "tasks（本次实际跑了哪些卡）—— 不在返回值里，靠调用方自己记",
                "max_chunk_len / segment_policy / max_steps / max_len —— 全部不在返回值里",
                "base_path / card_paths（用的哪份权重）—— 是 engine 属性，不进产物",
                "spec 快照（类别体系版本）—— 不进产物",
                "阈值 / 拒答判据 —— 不存在该概念（无输出即不发射，见 engine.py:164 pred_cls==0 break）",
            ],
        },
        "engine 关键属性": {"base_path": engine.base_path, "card_paths": engine.card_paths,
                            "attached": engine.attached},
        "锚点里有的字段": sorted({k for a in all_anchors for k in a}),
        "算子产物可溯源": "否：item = {label, start, end, score}，来源卡名 / step / class_id / color 全部丢弃",
    }

    out = {"text": TEXT, "structure": structure, "interval": interval,
           "dataset_semantics": dataset_semantics, "closure": closure, "trace": trace}
    Path("experiments/out_invariants/closure.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 键集")
    for k in ("引擎锚点键", "to_items 产物键", "filter 产物键", "pair 产物键"):
        print(f"  {k}: {closure[k]}")
    print("== 回喂结果")
    for k, v in closure.items():
        if k.startswith("回喂") or k.startswith("引擎锚点直喂") or k.startswith("decide"):
            print(f"  {k}: {'OK' if v['ok'] else 'FAIL'} {v.get('error','')}")
    print("== 区间语义（引擎）")
    for k, v in interval.items():
        for it in v:
            print(f"  {k}: [{it['s0']},{it['e0']}] 闭={it['闭区间切片']!r} 半开={it['半开区间切片']!r}")
    print("== 数据集真值语义", json.dumps(dataset_semantics, ensure_ascii=False))
    print("== 组合真实调用")
    print("  pair 产物:", json.dumps(combined, ensure_ascii=False))
    print("  decide:", decide(combined))
    print("== 组合真实调用 3（标签同源性）")
    print("  ", json.dumps(combo3, ensure_ascii=False))
    print("  ", json.dumps(combo2, ensure_ascii=False))
    print("== 顶层返回键", sorted(res.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
