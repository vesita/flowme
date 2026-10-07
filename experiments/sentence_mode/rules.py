#!/usr/bin/env python3
"""P7 `max_naive` 电池（口径 = PREREG §4，跑前写死；`fit=train`，逐出口）。

R0 majority / R1 punct(simple+full) / R2 particle / R3 final-char /
R4 len-bucket / R5 first-char / R6 labeldef；`max_naive = max(R0..R6)`。

产物：`results/battery.json`（含逐条 acc、seen_rate、以及 test 上的逐行预测
      —— 供 `analyze.py` 算「卡 − max_naive」的**配对** SE）。

用法：uv run python experiments/sentence_mode/rules.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import labels as L  # noqa: E402

DATA = HERE / "data"
#: 池内出口名 → 行内 `trig`/`exit` 的键（colloq 池的 keep/mask 就是 colloq / colloq_mask）
ename_key = {"keep": "keep", "maskfinal": "maskfinal", "mask": "mask",
             "colloq": "keep", "colloq_mask": "mask"}
RESULTS = HERE / "results"

RULE_KEYS = ("R0_majority", "R1_punct_simple", "R1_punct_full", "R2_particle",
             "R3_final_char", "R4_len_bucket", "R5_first_char", "R6_labeldef")

PARTICLE: tuple[str, ...] = (
    "吗", "呢", "吧", "难道", "是不是", "能不能", "有没有", "什么", "啥", "谁",
    "哪", "怎么", "咋", "多少", "几", "为什么", "如何", "呀", "啊",
)

#: (池, 出口名, 行内 exit 键) —— 口语池的 keep/mask 就是 colloq / colloq_mask
POOLS = (
    ("labeled", "keep", "keep"), ("labeled", "maskfinal", "maskfinal"),
    ("labeled", "mask", "mask"),
    ("colloq", "colloq", "keep"), ("colloq", "colloq_mask", "mask"),
)


def load(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def lookup(fit_keys, fit_y, eval_keys):
    """键 → train 多数类；未见键回退 train 全局多数（口径同 `free_rule_floor/rules.py`）。"""
    glob = Counter(fit_y).most_common(1)[0][0]
    tab: dict = defaultdict(Counter)
    for k, y in zip(fit_keys, fit_y):
        tab[k][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    seen = sum(1 for k in eval_keys if k in maj)
    return [maj.get(k, glob) for k in eval_keys], seen / max(1, len(eval_keys)), glob


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


def battery() -> dict:
    tr, te = load("train"), load("test")
    ctr, cte = load("colloq_train"), load("colloq_test")
    pool = {"labeled": (tr, te), "colloq": (ctr, cte)}
    out: dict = {"fit": "train", "particle_order": list(PARTICLE), "exits": {}}

    for pname, ename, ekey in POOLS:
        trr, ter = pool[pname]
        fit_s = [r["exit"][ekey] for r in trr]
        fit_y = [r["mode"] for r in trr]
        ev_s = [r["exit"][ekey] for r in ter]
        ev_y = [r["mode"] for r in ter]
        n = len(ev_y)
        maj = Counter(fit_y).most_common(1)[0][0]
        glob_acc = round(sum(1 for y in ev_y if y == maj) / n, 4)

        keys = {
            "R1_punct_simple": [k_punct(s) for s in fit_s], "R2_particle": [k_particle(s) for s in fit_s],
            "R3_final_char": [k_final(s) for s in fit_s],
            "R4_len_bucket": [len_bucket(len(s)) for s in fit_s],
            "R5_first_char": [k_first(s) for s in fit_s],
        }
        ev_keys = {k: [fn(s) for s in ev_s] for k, fn in (
            ("R1_punct_simple", k_punct), ("R2_particle", k_particle),
            ("R3_final_char", k_final), ("R4_len_bucket", lambda s: len_bucket(len(s))),
            ("R5_first_char", k_first))}

        rec: dict = {"n": n, "n_fit": len(fit_y),
                     "R0_majority": glob_acc, "majority_class": int(maj),
                     "majority_pct_fit": round(
                         sum(1 for y in fit_y if y == maj) / len(fit_y), 4),
                     "seen_rate": {}}
        preds: dict[str, list[int]] = {"R0_majority": [maj] * n}

        for k in keys:
            p, sr, _ = lookup(keys[k], fit_y, ev_keys[k])
            preds[k] = p
            rec[k] = round(sum(1 for a, b in zip(p, ev_y) if a == b) / n, 4)
            rec["seen_rate"][k] = round(sr, 4)

        # 触发词通道：单列披露的平凡基线（train 最常见的 trigger，含 ∅）
        def tg(r, k=ename_key[ename]):
            v = r["trig"].get(k)
            return tuple(v) if v else "∅"
        tri_tr = [tg(r) for r in trr]
        tri_ev = [tg(r) for r in ter]
        tri_maj = Counter(tri_tr).most_common(1)[0][0]
        rec["R_trig_majority_single"] = round(
            sum(1 for a, b in zip([tri_maj] * len(tri_ev), tri_ev) if a == b) / len(tri_ev), 4)
        rec["trig_none_pct_test"] = round(sum(1 for t in tri_ev if t == "∅") / len(tri_ev), 4)

        # R1 full / R6 labeldef：标签定义函数在该出口输入上的零拟合执行
        r6 = [L.MODE_ID[L.label_defn_on(s)] for s in ev_s]
        r6_fit = [L.MODE_ID[L.label_defn_on(s)] for s in fit_s]
        rec["R1_punct_full"] = round(sum(1 for a, b in zip(r6, ev_y) if a == b) / n, 4)
        rec["R6_labeldef"] = rec["R1_punct_full"]
        rec["R6_labeldef_fit"] = round(
            sum(1 for a, b in zip(r6_fit, fit_y) if a == b) / len(fit_y), 4)
        preds["R6_labeldef"] = r6

        # ---- M0 恒等式断言：E_mask 上 R-punct(simple) 必须 ≡ majority ----
        if ename in ("mask", "colloq_mask"):
            assert all(k_punct(s) == "∅" for s in ev_s + fit_s), \
                f"[M0 违例] {ename} 出口残留句末标点，遮蔽未生效"
            assert preds["R1_punct_simple"] == preds["R0_majority"], \
                f"[M0 违例] {ename} 上 R-punct(simple) ≠ majority"

        rules = RULE_KEYS
        best = max(rules, key=lambda k: rec[k])
        rec["max_naive"] = rec[best]
        rec["max_naive_rule"] = best
        rec["se_at_max"] = round(se(rec["max_naive"], n), 4)
        rec["one_minus_2se"] = round(1 - 2 * rec["se_at_max"], 4)
        rec["M3_rule_lt_1m2se"] = bool(rec["max_naive"] < rec["one_minus_2se"])
        out["exits"][ename] = {"pool": rec | {"rules": rules}, "pred": preds, "gold": ev_y}

    # 口径自检：R1_punct_full ≡ R6_labeldef
    for e, v in out["exits"].items():
        assert abs(v["pool"]["R1_punct_full"] - v["pool"]["R6_labeldef"]) < 1e-12
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "battery.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    out = battery()
    print(f"{'exit':12s} {'n':>6s} {'maj':>7s} {'R1s':>7s} {'R1f':>7s} {'R2p':>7s} "
          f"{'R3c':>7s} {'R4l':>7s} {'R5f':>7s} {'R6':>7s} {'max':>7s} {'best':>16s}")
    for e, v in out["exits"].items():
        r = v["pool"]
        print(f"{e:12s} {r['n']:6d} {r['R0_majority']:7.4f} {r['R1_punct_simple']:7.4f} "
              f"{r['R1_punct_full']:7.4f} {r['R2_particle']:7.4f} {r['R3_final_char']:7.4f} "
              f"{r['R4_len_bucket']:7.4f} {r['R5_first_char']:7.4f} {r['R6_labeldef']:7.4f} "
              f"{r['max_naive']:7.4f} {r['max_naive_rule']:>16s}")
    print("[M0] E_mask / E_colloq_mask 上 R-punct(simple) ≡ majority：断言通过")
    print(f"[done] results/battery.json")


if __name__ == "__main__":
    main()
