#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12c：补测 PREREG §3「不计入 E0」的四条披露规则（只读 data/*.jsonl，纯 CPU，不训练）。

四条披露项（PREREG §3 原文）：
  R-keyword（上下文显式同一性词）
  R-abbr   （手写简称词表：包含 + 后缀 大学/公司/网/社 + 去姓）
  R-prefix  （两提及紧邻前 2 字是否相同）
  R-ctxwin  （两侧 ±4 字窗口串是否相等）

口径与 P12a e0_literal_rules.py 完全一致（保证可比）：
  - 阈值族在 train 上取 acc 最高者 → test 评测；另报 test 上族内 oracle；两者取大 = 该规则分数；
  - 阈值网格 = train∪test 唯一值的中点 ± 哨兵（P12a REPORT §1③ 写死的同一网格）；
  - 门槛 GATE = 1 − 2×SE，test n=1200 ⇒ SE=sqrt(0.25/1200)=1.443pt ⇒ 97.114%。

「定义不清处按最有利方向取 oracle」= 故意给规则最大优势（全部写死在下面，跑前可见）：
  - R-keyword：关键词表取**并集最大化**（PREREG 禁词表 + 同义/指代词 + 称谓 + 类别词），
    计数与 0/1 命中各作候选，两个方向（命中⇒SAME / 命中⇒DIFF）都进族；
  - R-abbr：简写启发式**全列**（包含 / 子序列 / 后缀表 / 含与不含包含判定的后缀 / 去姓 /
    首 1/2/3 字同 / 去后缀回补），两个方向都进族；
  - R-prefix：把「前 2 字」的**两种读法都放进族**（提及串自身前缀 pf1..pf3、后缀 sf1..sf3、
    以及紧邻提及之前的上下文前缀 ctxpf1/2/4 —— PREREG「紧邻前 2 字」字面歧义，取最大者）；
  - R-ctxwin：窗口宽取 k=2/3/4 三种、前后/双侧/任一侧、含「提及自身在窗内」的包含读法、
    以及窗口字符集 Jaccard 分数族，两个方向都进族。
逐型（A/B/C/D 各 300）拆开报：因为风险是「某条对某一型满分」，合并 acc 会掩盖它。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ---- PREREG §4 常量（与 P12a 相同）----
N_TEST = 1200
N_TYPE = 300
SE = math.sqrt(0.25 / N_TEST)          # 0.0144338 → 1.443pt
SE_TYPE = math.sqrt(0.25 / N_TYPE)     # 0.0288675 → 2.887pt
GATE = 1 - 2 * SE                      # 0.9711324 → 97.114%
TYPES = ["A", "B", "C", "D"]

# ---------- 关键词表（R-keyword，最大优势：并集最大化）----------
KW_ID = ["简称", "又称", "同一人", "另一所", "两人", "即", "也就是",
         "同一", "同一所", "同一机构", "同一单位", "同一家", "亦称", "又名", "原名",
         "改名", "本名", "同名", "就是", "指的是", "此人", "该人", "本人", "前者", "后者"]
KW_CAT = ["公司", "大学", "学院", "中学", "医院", "学校", "研究院", "集团", "报社",
          "出版社", "中心", "局", "厂", "网", "社"]
KW_TITLE = ["先生", "女士", "医生", "教师", "研究员", "工程师", "律师", "记者",
            "设计师", "会计师", "编辑", "主管"]
KW_PRON = ["他", "她", "该校", "该", "此", "其"]

# ---------- 简称后缀表（R-abbr）----------
ABBR_SUF = ["大学", "学院", "中学", "学校", "医院", "公司", "集团", "网", "社", "大", "厂", "局"]


def cnt(text: str, kws) -> float:
    return float(sum(text.count(k) for k in kws))


def subseq(a: str, b: str) -> bool:
    """a 是否为 b 的子序列（取首字/取前字类启发式的最有利读法）。"""
    it = iter(b)
    return all(c in it for c in a)


def feats_kw(it: dict) -> dict:
    t = it["text"]
    return {
        "kw_id": cnt(t, KW_ID), "kw_cat": cnt(t, KW_CAT),
        "kw_title": cnt(t, KW_TITLE), "kw_pron": cnt(t, KW_PRON),
        "kw_id_any": 1.0 if cnt(t, KW_ID) > 0 else 0.0,
        "kw_cat_any": 1.0 if cnt(t, KW_CAT) > 0 else 0.0,
        "kw_title_any": 1.0 if cnt(t, KW_TITLE) > 0 else 0.0,
        "kw_pron_any": 1.0 if cnt(t, KW_PRON) > 0 else 0.0,
        "kw_any": 1.0 if cnt(t, KW_ID + KW_CAT + KW_TITLE + KW_PRON) > 0 else 0.0,
    }


def feats_abbr(it: dict) -> dict:
    s1, s2 = it["s1"], it["s2"]
    short, long_ = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    contain = 1.0 if short in long_ else 0.0
    return {
        "ab_contain": contain,
        "ab_subseq": 1.0 if subseq(s1, s2) or subseq(s2, s1) else 0.0,
        "ab_suf_in": 1.0 if contain and any(short.endswith(x) for x in ABBR_SUF) else 0.0,
        "ab_suf": 1.0 if any(short.endswith(x) for x in ABBR_SUF) else 0.0,
        "ab_desur": 1.0 if (len(s1) > 1 and s1[1:] == s2) or (len(s2) > 1 and s2[1:] == s1) else 0.0,
        "ab_dropsuf": 1.0 if (s1.rstrip("大学") == s2 or s2.rstrip("大学") == s1) else 0.0,
        "ab_f1": 1.0 if s1[:1] == s2[:1] else 0.0,
        "ab_f2": 1.0 if s1[:2] == s2[:2] else 0.0,
        "ab_f3": 1.0 if s1[:3] == s2[:3] else 0.0,
    }


def ctx_before(text: str, span, k: int) -> str:
    s = span[0]
    return text[max(0, s - k):s]


def feats_prefix(it: dict) -> dict:
    s1, s2, t = it["s1"], it["s2"], it["text"]
    f = {f"pf{k}": 1.0 if s1[:k] == s2[:k] else 0.0 for k in (1, 2, 3)}
    f.update({f"sf{k}": 1.0 if s1[-k:] == s2[-k:] else 0.0 for k in (1, 2, 3)})
    # 「两提及紧邻前 k 字」的上下文读法
    f.update({f"ctxpf{k}": 1.0 if ctx_before(t, it["m1_span"], k) == ctx_before(t, it["m2_span"], k) else 0.0
              for k in (1, 2, 4)})
    return f


def window(text: str, span, k: int, side: str) -> str:
    a, b = span
    if side == "b":
        return text[max(0, a - k):a]
    return text[b:b + k]


def jac(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def feats_ctxwin(it: dict) -> dict:
    t, m1, m2 = it["text"], it["m1_span"], it["m2_span"]
    f = {}
    for k in (2, 3, 4):
        b1, b2 = window(t, m1, k, "b"), window(t, m2, k, "b")
        a1, a2 = window(t, m1, k, "a"), window(t, m2, k, "a")
        f[f"cw_b{k}"] = 1.0 if b1 == b2 else 0.0
        f[f"cw_a{k}"] = 1.0 if a1 == a2 else 0.0
        f[f"cw_ba{k}"] = 1.0 if (b1 == b2 and a1 == a2) else 0.0
        f[f"cw_e{k}"] = 1.0 if (b1 == b2 or a1 == a2) else 0.0
        f[f"cw_jb{k}"] = jac(b1, b2)
        f[f"cw_ja{k}"] = jac(a1, a2)
    # 「邻近提及里找字面相同/包含关系」：s2 落在 m1 的 ±k 窗内（含提及自身）或反之
    for k in (2, 4):
        w1 = t[max(0, m1[0] - k):m1[1] + k]
        w2 = t[max(0, m2[0] - k):m2[1] + k]
        f[f"cw_in{k}"] = 1.0 if (it["s2"] in w1 or it["s1"] in w2) else 0.0
        # 排除提及自身占用的区间（更严格的「邻近」读法）
        x1 = t[max(0, m1[0] - k):m1[0]] + t[m1[1]:m1[1] + k]
        x2 = t[max(0, m2[0] - k):m2[0]] + t[m2[1]:m2[1] + k]
        f[f"cw_inx{k}"] = 1.0 if (it["s2"] in x1 or it["s1"] in x2) else 0.0
    return f


FAMILIES = {
    "R-keyword": feats_kw,
    "R-abbr": feats_abbr,
    "R-prefix": feats_prefix,
    "R-ctxwin": feats_ctxwin,
}


def load(split):
    p = DATA / f"{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def thresholds(vals):
    """与 P12a 同一网格：train∪test 唯一值的中点 ± 哨兵。"""
    u = sorted(set(vals))
    ts = [u[0] - 1.0]
    ts += [(u[i] + u[i + 1]) / 2 for i in range(len(u) - 1)]
    ts.append(u[-1] + 1.0)
    return ts


def acc(pred, y):
    return sum(1 for p, t in zip(pred, y) if p == t) / len(y)


def apply_cand(vals, d, t):
    return [1 if ((v <= t) if d == "le" else (v >= t)) else 0 for v in vals]


def per_type(pred, y, items):
    out = {}
    for ty in TYPES:
        idx = [i for i, it in enumerate(items) if it["type"] == ty]
        out[ty] = {"n": len(idx),
                   "acc": acc([pred[i] for i in idx], [y[i] for i in idx])}
    return out


def main():
    train, test = load("train"), load("test")
    ytr = [1 if it["label"] == "DIFF" else 0 for it in train]
    yte = [1 if it["label"] == "DIFF" else 0 for it in test]

    rows, all_pass = [], True
    for fam, ffn in FAMILIES.items():
        ftr = [ffn(it) for it in train]
        fte = [ffn(it) for it in test]
        names = list(ftr[0].keys())
        cands = []
        for mn in names:
            tr = [f[mn] for f in ftr]
            te = [f[mn] for f in fte]
            for d in ("le", "ge"):
                for t in thresholds(tr + te):
                    ptr = apply_cand(tr, d, t)
                    pte = apply_cand(te, d, t)
                    cands.append({"metric": mn, "dir": d, "t": t,
                                  "acc_tr": acc(ptr, ytr), "acc_te": acc(pte, yte),
                                  "_pte": pte, "_ptr": ptr})
        best_tr = max(cands, key=lambda c: c["acc_tr"])   # 平手 → 生成顺序首个（与 P12a 同）
        best_te = max(cands, key=lambda c: c["acc_te"])
        # 逐特征的族内 test 上限（追溯用：区分「PREREG 字面读法」与「最大优势读法」）
        feat_best = {}
        for mn in names:
            sub = [c for c in cands if c["metric"] == mn]
            b = max(sub, key=lambda c: c["acc_te"])
            feat_best[mn] = {"test_oracle": b["acc_te"], "dir": b["dir"], "t": b["t"],
                             "train_fit_test": max(sub, key=lambda c: c["acc_tr"])["acc_te"]}
        score = max(best_tr["acc_te"], best_te["acc_te"])
        score_cand = best_tr if best_tr["acc_te"] >= best_te["acc_te"] else best_te
        passed = score < GATE
        all_pass &= passed
        row = {
            "family": fam, "n_features": len(names), "n_candidates": len(cands),
            "train_fit": {"metric": best_tr["metric"], "dir": best_tr["dir"], "t": best_tr["t"],
                          "acc_train": best_tr["acc_tr"], "acc_test": best_tr["acc_te"],
                          "per_type_test": per_type(
                              apply_cand([f[best_tr["metric"]] for f in fte],
                                         best_tr["dir"], best_tr["t"]), yte, test)},
            "test_oracle": {"metric": best_te["metric"], "dir": best_te["dir"], "t": best_te["t"],
                            "acc_test": best_te["acc_te"],
                            "per_type_test": per_type(best_te["_pte"], yte, test)},
            "score": score, "score_source": ("train_fit" if score_cand is best_tr else "test_oracle"),
            "feature_best": feat_best,
            "score_per_type": per_type(score_cand["_pte"], yte, test),
            "se": SE, "se_type": SE_TYPE, "gate": GATE, "below_gate": passed,
            "per_type_100": [ty for ty, v in per_type(score_cand["_pte"], yte, test).items()
                             if v["acc"] >= 1.0],
        }
        rows.append(row)
        pt = row["score_per_type"]
        print(f"{fam}: feats={len(names)} n_cand={len(cands)} | "
              f"train拟合({best_tr['metric']},{best_tr['dir']},t={best_tr['t']:.4f}) "
              f"train={best_tr['acc_tr']:.4%}→test={best_tr['acc_te']:.4%} | "
              f"test oracle({best_te['metric']},{best_te['dir']},t={best_te['t']:.4f})="
              f"{best_te['acc_te']:.4%} | 取大={score:.4%} "
              f"({'<通过>' if passed else '≥未过(构念门不成立)'}) "
              f"| 逐型 {row['score_source']} "
              + " ".join(f"{ty}={pt[ty]['acc']:.1%}" for ty in TYPES))

    verdict = ("构念门仍成立（四条披露规则全 < 门槛）" if all_pass
               else "构念门不成立（有披露规则 ≥ 门槛）")
    out = {"n_train": len(train), "n_test": len(test), "se": SE, "se_type": SE_TYPE,
           "gate": GATE, "rules": rows, "all_below_gate": all_pass, "verdict": verdict}
    (HERE / "e0_disclosed_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"verdict: {verdict} (all_below_gate={all_pass})")
    raise SystemExit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
