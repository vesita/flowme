#!/usr/bin/env python3
"""P20 免费规则地板（**训练前先算**；PREREG §4，门槛 = 1 − 2SE）。

只读：experiments/sentence_mode/{data,labels.py,results/battery.json}、
      experiments/entity_identity/data、experiments/two_channel_head/data（不使用其标签做地板）。
只写：experiments/core_arch/results/floor.json。

用法：uv run python experiments/core_arch/rules_floor.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "sentence_mode"))

import data as D  # noqa: E402
import labels as L  # noqa: E402  只读导入

RESULTS = HERE / "results"
PARTICLE = ("吗", "呢", "吧", "难道", "是不是", "能不能", "有没有", "什么", "啥", "谁",
            "哪", "怎么", "咋", "多少", "几", "为什么", "如何", "呀", "啊")


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def se_worst(n: int) -> float:
    """PREREG §4 门槛口径：SE = sqrt(0.25/n)。"""
    return math.sqrt(0.25 / max(n, 1))


def lookup(fit_keys, fit_y, ev_keys):
    glob = Counter(fit_y).most_common(1)[0][0]
    tab: dict = defaultdict(Counter)
    for k, y in zip(fit_keys, fit_y):
        tab[k][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    seen = sum(1 for k in ev_keys if k in maj)
    return [maj.get(k, glob) for k in ev_keys], seen / max(1, len(ev_keys)), glob


def acc(pred, y):
    return round(sum(1 for a, b in zip(pred, y) if a == b) / len(y), 4)


def len_bucket(n: int) -> str:
    return "≤10" if n <= 10 else "≤20" if n <= 20 else "≤30" if n <= 30 else ">30"


def k_punct(s: str) -> str:
    t = s.rstrip()
    return t[-1] if t and t[-1] in L.MASK_SET else "∅"


def k_final(s: str) -> str:
    t = s.rstrip()
    return t[-1] if t else "∅"


def k_first(s: str) -> str:
    return s[0] if s else "∅"


def k_particle(s: str) -> str:
    for w in PARTICLE:
        if w in s:
            return w
    return "∅"


# ---------------------------------------------------------------------------
# Task S 地板（与 sentence_mode/rules.py 同口径，独立实现 + 对账）
# ---------------------------------------------------------------------------
def floor_S() -> dict:
    sm = D.load_sentence_mode()
    ref = json.loads((ROOT / "experiments" / "sentence_mode" / "results"
                      / "battery.json").read_text(encoding="utf-8"))
    out: dict = {"fit": "train", "particle_order": list(PARTICLE), "exits": {}}
    for ekey in ("keep", "mask"):
        fit_s = [r["exit"][ekey] for r in sm["train"]]
        fit_y = [r["mode"] for r in sm["train"]]
        ev_s = [r["exit"][ekey] for r in sm["test"]]
        ev_y = [r["mode"] for r in sm["test"]]
        n = len(ev_y)
        rules = {
            "R0_majority": ([Counter(fit_y).most_common(1)[0][0]] * n,
                            round(sum(1 for y in ev_y if y == Counter(fit_y).most_common(1)[0][0]) / n, 4)),
        }
        for name, fn in (("R1_punct_simple", k_punct), ("R2_particle", k_particle),
                         ("R3_final_char", k_final),
                         ("R4_len_bucket", lambda s: len_bucket(len(s))),
                         ("R5_first_char", k_first)):
            p, sr, _ = lookup([fn(s) for s in fit_s], fit_y, [fn(s) for s in ev_s])
            rules[name] = (p, acc(p, ev_y))
        r6 = [L.MODE_ID[L.label_defn_on(s)] for s in ev_s]
        rules["R6_labeldef"] = (r6, acc(r6, ev_y))
        rules["R1_punct_full"] = (r6, acc(r6, ev_y))   # 同一函数（口径同 sentence_mode）
        # 对账：R0/R2/R3/R4/R5 必须与既有 battery.json 逐位相等
        chk = ref["exits"][ekey]["pool"]
        parity = {}
        for name in ("R0_majority", "R2_particle", "R3_final_char", "R4_len_bucket",
                     "R5_first_char", "R6_labeldef"):
            mine, theirs = rules[name][1], chk[name]
            parity[name] = {"mine": mine, "theirs": theirs, "diff": round(abs(mine - theirs), 6)}
            assert abs(mine - theirs) <= 1e-9, f"对账失败 {ekey}/{name}: {mine} vs {theirs}"
        preds = {k: v[0] for k, v in rules.items()}
        scores = {k: v[1] for k, v in rules.items()}
        best = max(scores, key=scores.get)
        rec = {"n": n, "rules": scores, "seen_rate": None,
               "max_naive": scores[best], "max_naive_rule": best,
               "se_at_max": round(se(scores[best], n), 4),
               "one_minus_2se": round(1 - 2 * se_worst(n), 4),
               "one_minus_2se_at_floor": round(1 - 2 * se(scores[best], n), 4),
               "majority": scores["R0_majority"],
               "majority_plus_2se": round(scores["R0_majority"] + 2 * se(scores["R0_majority"], n), 4),
               "parity_vs_sentence_mode_battery": parity}
        rec["usable_rule_cannot_solve"] = bool(rec["max_naive"] < rec["one_minus_2se"])
        out["exits"][ekey] = rec
    return out


# ---------------------------------------------------------------------------
# Task T 地板（P12a 同口径；阈值 train 拟合）
# ---------------------------------------------------------------------------
def _edit(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _lcs(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _bigram(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s} if s else set()


def _jaccard(a: str, b: str) -> float:
    A, B = _bigram(a), _bigram(b)
    return len(A & B) / max(1, len(A | B))


def _share(a: str, b: str) -> float:
    return len(set(a) & set(b)) / max(1, max(len(set(a)), len(set(b))))


def floor_T() -> dict:
    tr, te = D.load_entity()["train"], D.load_entity()["test"]
    fit_y = [D.ent_label(r) for r in tr]
    ev_y = [D.ent_label(r) for r in te]
    n = len(ev_y)
    out: dict = {"fit": "train", "n_train": len(tr), "n_test": n, "rules": {}}

    def add(name, pred):
        out["rules"][name] = {"acc": acc(pred, ev_y), "train_acc": acc(pred, fit_y)}

    add("R0_majority", [Counter(fit_y).most_common(1)[0][0]] * n)
    for nm, fn in (("R_len_bucket", lambda r: len_bucket(len(r["text"]))),
                   ("R_first_char", lambda r: r["text"][0] if r["text"] else "∅"),
                   ("R_last_char", lambda r: r["text"][-1] if r["text"] else "∅"),
                   ("R_s1_eq_s2", lambda r: r["s1"] == r["s2"]),
                   ("R_s1_in_s2", lambda r: r["s1"] in r["s2"]),
                   ("R_s2_in_s1", lambda r: r["s2"] in r["s1"])):
        p, _, _ = lookup([fn(r) for r in tr], fit_y, [fn(r) for r in te])
        add(nm, p)

    # 阈值族：train∪test 唯一值中点，train 拟合
    def threshold_rule(name, scorer, higher_same: bool):
        vals = sorted({scorer(r) for r in tr} | {scorer(r) for r in te})
        cands = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        cands += [vals[0] - 1e-9, vals[-1] + 1e-9] if vals else []
        best_t, best_a, best_dir = None, -1.0, None
        for t in cands:
            for direc in (">=", "<="):
                pr = [(1 if scorer(r) >= t else 0) if direc == ">="
                      else (1 if scorer(r) <= t else 0) for r in tr]
                a = sum(1 for x, y in zip(pr, fit_y) if x == y) / len(fit_y)
                if a > best_a + 1e-12:
                    best_t, best_a, best_dir = t, a, direc
        pr = [(1 if scorer(r) >= best_t else 0) if best_dir == ">="
              else (1 if scorer(r) <= best_t else 0) for r in te]
        out["rules"][name] = {"acc": acc(pr, ev_y), "train_acc": round(best_a, 4),
                              "threshold": round(best_t, 6), "direction": best_dir}

    threshold_rule("R_edit_dist", lambda r: _edit(r["s1"], r["s2"]), False)
    threshold_rule("R_bigram_jaccard", lambda r: _jaccard(r["s1"], r["s2"]), True)
    threshold_rule("R_lcs_maxlen", lambda r: _lcs(r["s1"], r["s2"]) / max(1, max(len(r["s1"]), len(r["s2"]))), True)
    threshold_rule("R_share_ratio", lambda r: _share(r["s1"], r["s2"]), True)
    # P12a 四条字面规则对账（容差 0.1pt）
    p12a = {"R_s1_eq_s2": 75.0, "R_edit_dist": 87.5, "R_bigram_jaccard": 86.667,
            "R_share_ratio": 87.5}
    parity = {}
    for k, v in p12a.items():
        mine = out["rules"][k]["acc"] * 100
        parity[k] = {"mine_pct": round(mine, 3), "p12a_pct": v, "diff_pt": round(mine - v, 3)}
        assert abs(mine - v) <= 0.1, f"P12a 对账失败 {k}: {mine} vs {v}"
    out["parity_vs_p12a"] = parity
    # 披露（不入地板）
    pv, _, _ = lookup([r["variant"] for r in tr], fit_y, [r["variant"] for r in te])
    out["disclosure_variant_not_in_floor"] = {"acc": acc(pv, ev_y)}

    floor = max(v["acc"] for v in out["rules"].values())
    best = max(out["rules"], key=lambda k: out["rules"][k]["acc"])
    out["max_naive"] = floor
    out["max_naive_rule"] = best
    out["se_at_max"] = round(se(floor, n), 4)
    out["one_minus_2se"] = round(1 - 2 * se_worst(n), 4)
    out["one_minus_2se_at_floor"] = round(1 - 2 * se(floor, n), 4)
    out["majority"] = out["rules"]["R0_majority"]["acc"]
    out["majority_plus_2se"] = round(out["majority"] + 2 * se(out["majority"], n), 4)
    out["usable_rule_cannot_solve"] = bool(floor < out["one_minus_2se"])
    return out


def main() -> None:
    rep = {"S": floor_S(), "T": floor_T()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "floor.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    for e, r in rep["S"]["exits"].items():
        print(f"[S/{e}] n={r['n']} max_naive={r['max_naive']} ({r['max_naive_rule']}) "
              f"maj={r['majority']} 1-2SE={r['one_minus_2se']} 可用={r['usable_rule_cannot_solve']} "
              f"对账=OK", flush=True)
    t = rep["T"]
    print(f"[T]     n={t['n_test']} max_naive={t['max_naive']} ({t['max_naive_rule']}) "
          f"maj={t['majority']} 1-2SE={t['one_minus_2se']} 可用={t['usable_rule_cannot_solve']} "
          f"P12a对账=OK 披露variant={t['disclosure_variant_not_in_floor']}", flush=True)
    print("[done] results/floor.json", flush=True)


if __name__ == "__main__":
    main()
