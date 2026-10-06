"""拒答出口实测（纯推理）—— 5 张卡 × {组 A 分布内背景, 组 B 分布外} 的开火率。

判据全部预注册在 PREREG.md（跑前写死）。本脚本不训练、不改 src/。

    uv run python experiments/out_invariants/run_reject.py 2>&1 | tee experiments/out_invariants/logs/run_reject.log
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402

CARDS = ["pronoun", "sentiment", "relation", "person", "negation"]
WEIGHTS = {
    "pronoun": "checkpoints/cards/pronoun.pt",
    "sentiment": "checkpoints/cards/sentiment.pt",
    "relation": "checkpoints/cards/relation.pt",
    "person": "checkpoints/cards/person.pt",
    "negation": "checkpoints/negation_accept_card_e12.pt",
}
BASE = "checkpoints/base_encoder.pt"
CARDS_DIR = "checkpoints/cards"

VAL_SEED = 42
BUILD_SAMPLES = 6000
N_A = 200      # 组 A 上限（PREREG §3）
N_B1 = 50      # 每张"其它卡"的交叉背景条数
N_B2 = 200
N_B3 = 200

BUCKETS = [("<0.5", -1.0, 0.5), ("0.5", 0.5, 0.6), ("0.6", 0.6, 0.7), ("0.7", 0.7, 0.8),
           ("0.8", 0.8, 0.9), ("0.9", 0.9, 1.01)]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# L2 已知答案对照（手挑，PREREG §4）
KNOWN = [
    ("pronoun_fire", "pronoun", True, "我今天特别高兴，谢谢你一直陪着我们。"),
    ("sentiment_fire", "sentiment", True, "我今天真的特别高兴。"),
    ("relation_fire", "relation", True, "老师让我们区分一目了然和不言而喻。"),
    ("person_fire", "person", True, "张三昨天把文件交给了李四，李四又转交给了王五。"),
    ("negation_fire", "negation", True, "他没有来开会，大家都在等他。"),
    ("pronoun_no", "pronoun", False, "桌子上放着三本书，书是蓝色的。"),
    ("sentiment_no", "sentiment", False, "会议定在周三上午九点，地点是三号会议室。"),
    ("relation_no", "relation", False, "今天天气很好，我们一起去公园散步。"),
    ("person_no", "person", False, "桌子上放着三本书，书是蓝色的。"),
    ("negation_no", "negation", False, "会议定在周三上午九点，地点是三号会议室。"),
]
MANUAL3 = ["他没有来开会，我特别高兴。", "老师让我们区分一目了然和不言而喻。",
           "会议定在周三上午九点，地点是三号会议室。"]


def fire(anchors: list[dict]) -> bool:
    """开火 = 该样本上吐了 ≥1 条切片（PREREG §2）。"""
    return bool(anchors)


def h(res: dict) -> str:
    return hashlib.sha256(
        json.dumps(res, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def val_split(card, samples: int, seed: int = VAL_SEED):
    """沿用 experiments/core_branch/eval_old_cards_branch.py:val_split 的口径。"""
    data = card.build_dataset(samples)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[:n_val]


def main() -> int:
    t_start = time.time()
    import os
    os.chdir(ROOT)

    engine = MultiTaskEngine(base_path=BASE, cards_dir=CARDS_DIR)
    engine.attach(WEIGHTS["negation"])
    print(f"[engine] base={engine.base_path} device={engine.device}")
    for k in CARDS:
        print(f"[engine] card={k} weight={engine.card_paths.get(k)}")
    missing = [k for k in CARDS if k not in engine.decoders]
    if missing:
        raise SystemExit(f"卡没挂上：{missing}")

    cards = resolve_tasks(CARDS)
    vals = {k: val_split(cards[k], BUILD_SAMPLES) for k in CARDS}
    bgs = {k: [s for s in vals[k] if not s.get("spans")] for k in CARDS}
    print("[data] " + " ".join(f"{k}: val={len(vals[k])} bg={len(bgs[k])}" for k in CARDS))

    # ---- 分组 -------------------------------------------------------------
    groups: dict[str, dict[str, list[str]]] = {}
    for k in CARDS:
        a = [s["text"] for s in bgs[k]][:N_A]
        b1: list[str] = []
        for other in CARDS:
            if other == k:
                continue
            b1 += [s["text"] for s in bgs[other]][:N_B1]
        groups[k] = {"A": a, "B1": b1}

    adv_route = [json.loads(l) for l in
                 open(ROOT / "experiments/adversarial_routing/adversarial.jsonl", encoding="utf-8")]
    adv_type = [json.loads(l) for l in
                open(ROOT / "experiments/input_type/adversarial.jsonl", encoding="utf-8")]
    for k in CARDS:
        b2 = [d["text"] for d in adv_route if d.get("true_label") != k][:N_B2]
        b3 = [d["text"] for d in adv_type][:N_B3]
        groups[k]["B2"] = b2
        groups[k]["B3"] = b3
        groups[k]["B"] = groups[k]["B1"] + b2 + b3
    print("[data] " + " ".join(
        f"{k}: A={len(groups[k]['A'])} B1={len(groups[k]['B1'])} "
        f"B2={len(groups[k]['B2'])} B3={len(groups[k]['B3'])}" for k in CARDS))

    # ---- L1 机械对照 ------------------------------------------------------
    l1 = {"empty_is_not_fire": fire([]) is False,
          "single_is_fire": fire([{"category": "x"}]) is True}
    # ---- L2 已知答案 ------------------------------------------------------
    known_res = []
    for kid, card, expect, text in KNOWN:
        r = engine.predict(text, tasks=[card])
        got = fire(r["tasks"][card])
        known_res.append({"id": kid, "card": card, "expect_fire": expect,
                          "got_fire": got, "ok": got == expect, "n_anchors": len(r["tasks"][card])})

    # ---- 测量（两遍 ⇒ 确定性） -------------------------------------------
    def run(card: str, text: str):
        return engine.predict(text, tasks=[card])

    rows: dict[str, dict] = {}
    det_mismatch: list[dict] = []
    for k in CARDS:
        rows[k] = {}
        for gname, texts in groups[k].items():
            out = []
            for t in texts:
                r1 = run(k, t)
                r2 = run(k, t)
                if h(r1) != h(r2):
                    det_mismatch.append({"card": k, "group": gname, "text": t[:40],
                                         "h1": h(r1), "h2": h(r2)})
                anchors = r1["tasks"][k]
                out.append({"text": t, "fire": fire(anchors), "n": len(anchors),
                            "max_conf": max((a["confidence"] for a in anchors), default=None),
                            "cats": sorted({a["class_name"] for a in anchors})})
            rows[k][gname] = out

    # ---- 汇总 -------------------------------------------------------------
    def rate(k, g):
        xs = rows[k][g]
        return (sum(x["fire"] for x in xs) / len(xs)) if xs else None

    table = {}
    for k in CARDS:
        a, b = rate(k, "A"), rate(k, "B")
        table[k] = {
            "n_A": len(groups[k]["A"]), "n_B": len(groups[k]["B"]),
            "fire_A": round(a, 4), "fire_B": round(b, 4),
            "delta": round(b - a, 4),
            "fire_B1": round(rate(k, "B1"), 4), "fire_B2": round(rate(k, "B2"), 4),
            "fire_B3": round(rate(k, "B3"), 4),
            "P1_pass": (b is not None and a is not None and b <= a + 0.10),
            "P3_pass": (b is not None and b <= 0.85),
        }
        print(f"[table] {k}: A={table[k]['fire_A']} B={table[k]['fire_B']} "
              f"d={table[k]['delta']} P1={table[k]['P1_pass']} P3={table[k]['P3_pass']}")

    # 置信度分桶 + 阈值扫描（阈值扫描 = 未预注册探索）
    calib, scan = {}, {}
    for k in CARDS:
        calib[k] = {}
        for gname in ("A", "B"):
            xs = rows[k][gname]
            fired = [x for x in xs if x["fire"] and x["max_conf"] is not None]
            bk = {}
            for name, lo, hi in BUCKETS:
                sel = [x for x in fired if lo <= x["max_conf"] < hi]
                bk[name] = {
                    "n": len(sel),
                    "share_of_group": round(len(sel) / len(xs), 4) if xs else 0.0,
                    "mean_conf": round(sum(x["max_conf"] for x in sel) / len(sel), 4) if sel else None,
                }
            bk["no_fire"] = {"n": len(xs) - len(fired),
                             "share_of_group": round((len(xs) - len(fired)) / len(xs), 4) if xs else 0.0,
                             "mean_conf": None}
            calib[k][gname] = bk
        scan[k] = {}
        for t in THRESHOLDS:
            def fr(xs, t=t):
                if not xs:
                    return None
                return round(sum(1 for x in xs if x["fire"] and (x["max_conf"] or 0) >= t) / len(xs), 4)
            scan[k][f"t>={t}"] = {"fire_A": fr(rows[k]["A"]), "fire_B": fr(rows[k]["B"])}

    p1_all = all(table[k]["P1_pass"] for k in CARDS)
    p3_all = all(table[k]["P3_pass"] for k in CARDS)
    worst = max(CARDS, key=lambda k: table[k]["delta"])
    l2_ok = all(x["ok"] for x in known_res if x["expect_fire"]) and \
        sum(x["ok"] for x in known_res) >= 8
    l1_ok = all(l1.values())
    det_ok = not det_mismatch

    verdict = "拒答成立" if (p1_all and p3_all and l1_ok and l2_ok and det_ok) else \
              ("证据不足" if not (l1_ok and l2_ok and det_ok) else "拒答不成立（分布外照样开火）")

    result = {
        "prereg": "experiments/out_invariants/PREREG.md",
        "base": BASE, "cards_dir": CARDS_DIR, "weights": WEIGHTS,
        "device": str(engine.device),
        "attached": engine.attached,
        "card_paths": engine.card_paths,
        "val_seed": VAL_SEED, "build_samples": BUILD_SAMPLES,
        "group_sizes": {k: {g: len(v) for g, v in groups[k].items()} for k in CARDS},
        "table": table,
        "P1_all": p1_all, "P3_all": p3_all, "worst_card": worst,
        "worst_delta": table[worst]["delta"],
        "determinism": {"mismatches": len(det_mismatch), "ok": det_ok,
                        "samples_checked": sum(len(groups[k][g]) for k in CARDS for g in groups[k])},
        "known_answers": {"L1": l1, "L1_ok": l1_ok, "L2": known_res,
                          "L2_ok": l2_ok,
                          "L2_fire_all_ok": all(x["ok"] for x in known_res if x["expect_fire"]),
                          "L2_score": f"{sum(x['ok'] for x in known_res)}/{len(known_res)}"},
        "calibration": calib,
        "threshold_scan_EXPLORATORY": scan,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    Path("experiments/out_invariants/results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] {verdict}  P1_all={p1_all} P3_all={p3_all} L1={l1_ok} L2={l2_ok} "
          f"det={det_ok} worst={worst} Δ={table[worst]['delta']} "
          f"elapsed={result['elapsed_sec']}s")

    # L1 第三项：3 条真实返回的原始 JSON 落盘，供人工清点
    raw = {t: engine.predict(t, tasks=CARDS) for t in MANUAL3}
    Path("experiments/out_invariants/raw").mkdir(exist_ok=True)
    Path("experiments/out_invariants/raw/manual_count3.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("experiments/out_invariants/raw/rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
