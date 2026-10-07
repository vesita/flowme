#!/usr/bin/env python3
"""P8 句式模式 → 骨架子集的**两套映射**（共享逻辑 + 映射落盘）。

- 映射 A（规则·骨架字面）：PREREG §1.1，程序化、确定性、与数据分布无关；
- 映射 B（统计·train）：PREREG §1.3，按 `g(sent)` 掩码等价类分组，
  组内 gold 骨架按频次降序取覆盖 ≥ X=0.90 的最短前缀；
- 行级句式标签 g(sent)：PREREG §1.2（**规则标签** —— `sentence_mode/` 并行单元产物未落盘）。

用法：uv run python experiments/mode_conditioned_skel/modes.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
RESULTS = HERE / "results"

MODES = ("陈述", "疑问", "祈使", "感叹", "反问")
#: 掩码等价类（PREREG §1.2 写死）：反问并入疑问；感叹子集为空 ⇒ 回退不掩码
MASK_ALIAS = {"陈述": "陈述", "疑问": "疑问", "祈使": "祈使", "反问": "疑问", "感叹": "感叹"}
FANWEN = ("难道", "岂", "莫非")
X_COVER = 0.90

_TAIL = "。．.…～~）)」』\"'”’、，,；;：:·"


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def skeleton_table() -> dict[int, str]:
    st = json.loads((TCH_DATA / "stats.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in st["skeletons"].items()}


# ---------------------------------------------------------------------------
# 映射 A：骨架字面 → 句式（PREREG §1.1，优先级自上而下）
# ---------------------------------------------------------------------------
def mode_of_skel(tpl: str) -> str:
    if any(k in tpl for k in FANWEN) or ("不是" in tpl and "吗" in tpl):
        return "反问"
    if "！" in tpl or "!" in tpl:
        return "感叹"
    if "吗" in tpl or "么" in tpl:
        return "疑问"
    if "呢" in tpl:
        return "疑问"
    if "吧" in tpl:
        return "祈使"
    return "陈述"


# ---------------------------------------------------------------------------
# 行级句式标签 g(sent)（PREREG §1.2，规则标签）
# ---------------------------------------------------------------------------
def mode_of_sent(sent: str) -> str:
    s = sent.rstrip()
    while s and s[-1] in _TAIL:
        s = s[:-1]
    if not s:
        return "陈述"
    c = s[-1]
    fanwen = any(k in sent for k in FANWEN) or ("不是" in sent and "吗" in sent)
    if c in "吗么呢":
        return "反问" if fanwen else "疑问"
    if c == "吧":
        return "祈使"
    if c in "！!":
        return "感叹"
    return "陈述"


def subsets_A(tab: dict[int, str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {m: [] for m in MODES}
    for sid, tpl in sorted(tab.items()):
        out[mode_of_skel(tpl)].append(sid)
    return out


# ---------------------------------------------------------------------------
# 映射 B：train 上按掩码等价类分组，取覆盖 ≥ X 的最短前缀
# ---------------------------------------------------------------------------
def subsets_B(train: list[dict], tab: dict[int, str], x: float = X_COVER) -> dict[str, list[int]]:
    by: dict[str, Counter] = defaultdict(Counter)
    for r in train:
        by[MASK_ALIAS[mode_of_sent(r["sent"])]][r["skel_id"]] += 1
    out: dict[str, list[int]] = {}
    for m, cnt in by.items():
        order = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        acc, keep = 0, []
        tot = sum(cnt.values())
        for sid, c in order:
            keep.append(sid)
            acc += c
            if acc / tot >= x:
                break
        out[m] = sorted(keep)
    for m in MODES:
        out.setdefault(MASK_ALIAS[m], out.get(MASK_ALIAS[m], []))
    return out


def cover(sub: dict[str, list[int]], rows: list[dict], mode_fn) -> float:
    """gold 落进对应子集的比例（mode_fn 给出行级句式）。"""
    if not rows:
        return float("nan")
    ok = 0
    for r in rows:
        m = MASK_ALIAS[mode_fn(r)]
        s = sub.get(m, [])
        if not s:                      # 空子集 ⇒ 不掩码 ⇒ 视为覆盖
            ok += 1
        elif r["skel_id"] in s:
            ok += 1
    return ok / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    tab = skeleton_table()
    assert len(tab) == 40, len(tab)
    sA = subsets_A(tab)
    train = load_rows(TCH_DATA / "train.jsonl")
    if a.smoke:
        train = train[:200]
    sB = subsets_B(train, tab)

    # ---- 逐骨架归属 + A/B 一致 ----
    per = []
    for sid, tpl in sorted(tab.items()):
        mA = mode_of_skel(tpl)
        hits = [m for m, ids in sB.items() if sid in ids]
        per.append({"id": sid, "tpl": tpl, "A": mA,
                    "B": hits,
                    "A_eq_B": bool(hits) and MASK_ALIAS[mA] in hits,
                    "n_train": sum(1 for r in train if r["skel_id"] == sid),
                    "g_train": dict(Counter(mode_of_sent(r["sent"])
                                            for r in train if r["skel_id"] == sid))})
    covered = [p for p in per if p["B"]]
    agree = sum(1 for p in covered if p["A_eq_B"])

    # ---- 各出口的 gold 覆盖率（A / B 两套）----
    splits = {"train": train,
              "test": load_rows(TCH_DATA / "test.jsonl"),
              "adv2": load_rows(ROOT / "experiments" / "struct_supervision" / "data" / "adv2.jsonl"),
              "a_bal": load_rows(ROOT / "experiments" / "skeleton_leak" / "data" / "a_bal.jsonl")}
    if a.smoke:
        for k in ("test", "adv2", "a_bal"):
            splits[k] = splits[k][:100]

    out = {
        "X": X_COVER, "modes": list(MODES),
        "table_size": len(tab),
        "subsets_A": {m: sA[m] for m in MODES},
        "size_A": {m: len(sA[m]) for m in MODES},
        "subsets_B": {m: sB.get(m, []) for m in sorted(sB)},
        "size_B": {m: len(sB.get(m, [])) for m in sorted(sB)},
        "per_skeleton": per,
        "agreement": {"n_skel": len(per), "n_B_covered": len(covered),
                      "n_B_multi": sum(1 for p in per if len(p["B"]) > 1),
                      "n_B_uncovered": sum(1 for p in per if not p["B"]),
                      "agree": agree,
                      "agree_rate_over_covered": round(agree / max(1, len(covered)), 4)},
        "mode_dist": {},
        "cover": {},
    }
    for name, rows in splits.items():
        cm = Counter(mode_of_sent(r["sent"]) for r in rows)
        cg = Counter(MASK_ALIAS[mode_of_skel(tab[r["skel_id"]])] for r in rows)
        out["mode_dist"][name] = {"n": len(rows), "by_g_sent": dict(cm),
                                  "by_gold_skel": dict(cg)}
        out["cover"][name] = {
            "A_oracle": round(cover(sA, rows, lambda r: MASK_ALIAS[mode_of_skel(tab[r["skel_id"]])]), 4),
            "A_rule": round(cover(sA, rows, lambda r: mode_of_sent(r["sent"])), 4),
            "B_oracle": round(cover(sB, rows, lambda r: MASK_ALIAS[mode_of_skel(tab[r["skel_id"]])]), 4),
            "B_rule": round(cover(sB, rows, lambda r: mode_of_sent(r["sent"])), 4),
        }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mapping.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[A] 子集大小 = {out['size_A']}")
    print(f"[B] 子集大小 = {out['size_B']}")
    print(f"[A/B] 覆盖 {len(covered)}/40、多归属 {out['agreement']['n_B_multi']}、"
          f"未覆盖 {out['agreement']['n_B_uncovered']}、一致 {agree}"
          f"（一致率 {out['agreement']['agree_rate_over_covered']}）")
    for k, v in out["mode_dist"].items():
        print(f"  [{k}] g={v['by_g_sent']} gold={v['by_gold_skel']}")
    print(f"[cover] {out['cover']}")
    print(f"[done] → {RESULTS / 'mapping.json'}")


if __name__ == "__main__":
    main()
