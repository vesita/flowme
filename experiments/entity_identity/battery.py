#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12b `max_naive` 完整电池（fit=train，逐条报 acc；PREREG §3 + D2 计数类）。

计入 max_naive 的规则（9 条，逐条报）：
  1 R-same   提及对字面相同 ⇒ DIFF（=PREREG R-exact 方向之一）
  2 R-edit   编辑距离（绝对/归一）阈值族
  3 R-ngram  字符 1/2-gram Jaccard 阈值族
  4 R-lcs    最长公共子串比 / 共享字比 / 包含判定 阈值族
  5 majority train 多数类
  6 length   文本长度桶 len//8 → train 该桶多数类
  7 structure (同句?, 提及数, 句距桶) → train 查表多数（PREREG §3 原样）
  8 n_cue    **计数类（D2）**：(m1句位, m2句位, 字距//4, 提及对是否含代称) → train 查表多数
  9 R-pron   提及对含代称（他/她/该校）⇒ SAME，否则 DIFF（可构造的第 1 条额外规则）

阈值族口径同 P12a e0_literal_rules.py：候选 = train∪test 唯一值中点 ± 哨兵；
**族内 train 拟合取 acc 最高**（平手取生成序首个），在 test 上评测。查表类无 min_support、
未见键回退 train 全局多数（口径同 free_rule_floor/rules.py::lookup）。

输出：results/battery.json + 逐条 stdout。纯 CPU。
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import e0_literal_rules as E0

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"
PRON = {"他", "她", "该校"}

ARMS = {"x": ("train.jsonl", "test.jsonl"),
        "x-ir": ("xir_train.jsonl", "xir_test.jsonl"),
        "x-cross": ("xcross_train.jsonl", "xcross_test.jsonl")}


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sent_idx(text: str, pos: int) -> int:
    return text[:pos].count("。")


def dist_bucket(a: int, b: int) -> int:
    return abs(a - b) // 4


def majority_pred(fit_y, eval_n):
    g = Counter(fit_y).most_common(1)[0][0]
    return [g] * eval_n


def lookup(fit_keys, fit_y, eval_keys):
    glob = Counter(fit_y).most_common(1)[0][0]
    tab = defaultdict(Counter)
    for k, y in zip(fit_keys, fit_y):
        tab[k][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    seen = sum(1 for k in eval_keys if k in maj)
    return [maj.get(k, glob) for k in eval_keys], seen / max(1, len(eval_keys))


def acc(pred, y):
    return sum(1 for p, t in zip(pred, y) if p == t) / len(y)


def threshold_family(name, mtr, mte, ytr, yte):
    """阈值族：train 拟合取 acc 最高 → test 评测（同 P12a）；连带返回逐条预测。"""
    cands = []
    for d in ("le", "ge"):
        for t in E0.thresholds(list(mtr) + list(mte)):
            ptr = [1 if ((v <= t) if d == "le" else (v >= t)) else 0 for v in mtr]
            pte = [1 if ((v <= t) if d == "le" else (v >= t)) else 0 for v in mte]
            cands.append({"acc_train": acc(ptr, ytr), "acc_test": acc(pte, yte),
                          "metric": name, "dir": d, "t": t, "pred_te": pte})
    best = max(cands, key=lambda c: c["acc_train"])
    best = dict(best)
    best["n_cand"] = len(cands)
    best["oracle"] = {c["metric"]: c["acc_test"] for c in cands if c["dir"] == best["dir"]}
    return best


def run_arm(arm: str) -> dict:
    tr = load(DATA / ARMS[arm][0])
    te = load(DATA / ARMS[arm][1])
    ytr = [1 if r["label"] == "DIFF" else 0 for r in tr]
    yte = [1 if r["label"] == "DIFF" else 0 for r in te]
    n_te = len(yte)
    se = math.sqrt(0.25 / n_te)

    mtr = [E0.metrics(r["s1"], r["s2"]) for r in tr]
    mte = [E0.metrics(r["s1"], r["s2"]) for r in te]
    rules: dict = {}
    preds: dict = {}

    # 1 字面相同（两方向都入族，train 拟合取高者 —— 同 P12a 两方向口径）
    s_tr = [int(m["same"] > 0.5) for m in mtr]
    s_te = [int(m["same"] > 0.5) for m in mte]
    d_diff = {"acc": acc(s_tr, ytr), "acc_test": acc(s_te, yte), "fit": "字面相同⇒DIFF"}
    d_same = {"acc": acc([1 - v for v in s_tr], ytr),
              "acc_test": acc([1 - v for v in s_te], yte), "fit": "字面相同⇒SAME"}
    if d_diff["acc"] >= d_same["acc"]:
        rules["R-same"], preds["R-same"] = d_diff, s_te
    else:
        rules["R-same"], preds["R-same"] = d_same, [1 - v for v in s_te]
    # 2-4 字面阈值族
    for fam, names in (("R-edit", ("edit_abs", "edit_norm")),
                       ("R-ngram", ("ng1", "ng2")),
                       ("R-lcs", ("lcs_ratio", "share_ratio", "contains"))):
        cands = []
        for nm in names:
            cands.append(threshold_family(nm, [m[nm] for m in mtr], [m[nm] for m in mte],
                                          ytr, yte))
        btr = max(cands, key=lambda c: c["acc_train"])
        preds[fam] = btr["pred_te"]
        rules[fam] = {"acc": btr["acc_train"], "acc_test": btr["acc_test"],
                      "fit": f"train 拟合 {btr['metric']}/{btr['dir']}/t={btr['t']:.4f}",
                      "oracle_test": {c["metric"]: c["acc_test"] for c in cands}}

    # 5 majority
    g = Counter(ytr).most_common(1)[0][0]
    preds["majority"] = [g] * len(yte)
    rules["majority"] = {"acc": acc([g] * len(ytr), ytr), "acc_test": acc(preds["majority"], yte),
                         "fit": f"train 多数类={g}"}
    # 6 长度桶
    ktr = [r["text_len"] // 8 for r in tr]
    kte = [r["text_len"] // 8 for r in te]
    p, seen = lookup(ktr, ytr, kte)
    preds["length"] = p
    rules["length"] = {"acc": acc(lookup(ktr, ytr, ktr)[0], ytr), "acc_test": acc(p, yte),
                       "fit": "len//8 → train 桶多数", "seen_rate": seen}
    # 7 structure（PREREG §3 原样）
    def sk(r):
        a = r["m1_span"][0]
        b = r["m2_span"][0]
        ia, ib = sent_idx(r["text"], a), sent_idx(r["text"], b)
        return (ia == ib, r["n_mentions"], abs(ia - ib))
    s_tr = [sk(r) for r in tr]
    s_te = [sk(r) for r in te]
    p, seen = lookup(s_tr, ytr, s_te)
    preds["structure"] = p
    rules["structure"] = {"acc": acc(lookup(s_tr, ytr, s_tr)[0], ytr), "acc_test": acc(p, yte),
                          "fit": "(同句?, 提及数, 句距桶) → train 查表多数", "seen_rate": seen}
    # 8 n_cue（D2 计数类）
    def ck(r):
        a, b = r["m1_span"][0], r["m2_span"][0]
        has_pron = int(bool({r["s1"], r["s2"]} & PRON))
        return (sent_idx(r["text"], a), sent_idx(r["text"], b),
                dist_bucket(a, b), has_pron)
    c_tr = [ck(r) for r in tr]
    c_te = [ck(r) for r in te]
    p, seen = lookup(c_tr, ytr, c_te)
    preds["n_cue"] = p
    rules["n_cue"] = {"acc": acc(lookup(c_tr, ytr, c_tr)[0], ytr), "acc_test": acc(p, yte),
                      "fit": "(m1句位, m2句位, 字距//4, 对含代称) → train 查表多数",
                      "seen_rate": seen}
    # 8b 计数类拆项（披露：只看句位 / 只看字距），便于读出哪条计数占掉
    for nm, fn in (("n_cue@句位", lambda r: (sent_idx(r["text"], r["m1_span"][0]),
                                             sent_idx(r["text"], r["m2_span"][0]))),
                   ("n_cue@字距桶", lambda r: dist_bucket(r["m1_span"][0], r["m2_span"][0])),
                   ("n_cue@m2句位", lambda r: sent_idx(r["text"], r["m2_span"][0]))):
        a_tr = [fn(r) for r in tr]
        a_te = [fn(r) for r in te]
        p, seen = lookup(a_tr, ytr, a_te)
        rules[nm] = {"acc": acc(lookup(a_tr, ytr, a_tr)[0], ytr),
                     "acc_test": acc(p, yte), "fit": f"{nm} → train 查表多数（披露项）",
                     "seen_rate": seen, "disclosed": True}
    # 9 R-pron
    pr = [int(bool({r["s1"], r["s2"]} & PRON)) for r in tr]
    pe = [int(bool({r["s1"], r["s2"]} & PRON)) for r in te]
    # 方向由 train 拟合
    direct = acc([1 - v for v in pr], ytr)   # 含代称 ⇒ DIFF 的反向
    direct_e = acc([1 - v for v in pe], yte)
    flip = acc(pr, ytr)
    flip_e = acc(pe, yte)
    if flip >= direct:
        preds["R-pron"] = pe
        rules["R-pron"] = {"acc": flip, "acc_test": flip_e, "fit": "含代称 ⇒ DIFF"}
    else:
        preds["R-pron"] = [1 - v for v in pe]
        rules["R-pron"] = {"acc": direct, "acc_test": direct_e, "fit": "含代称 ⇒ SAME"}

    counted = [k for k in rules if not rules[k].get("disclosed")]
    mx = max(counted, key=lambda k: rules[k]["acc_test"])
    out = {"arm": arm, "n_train": len(ytr), "n_test": n_te, "se": se,
           "rules": rules, "max_naive": rules[mx]["acc_test"], "max_naive_rule": mx,
           "counted_rules": counted, "preds": preds, "y": yte,
           "type": [r["type"] for r in te], "subtype": [r.get("subtype", r["type"]) for r in te]}
    print(f"\n=== {arm}（n_test={n_te}, SE={se:.4f}）max_naive = "
          f"{out['max_naive']:.4%}（{mx}）===")
    for k, v in rules.items():
        tag = "披露" if v.get("disclosed") else "计入"
        print(f"  [{tag}] {k:12s} train_acc={v['acc']:.4%} test_acc={v['acc_test']:.4%}"
              + (f" seen={v['seen_rate']:.3f}" if "seen_rate" in v else ""))
    return out


def main() -> int:
    RES.mkdir(exist_ok=True)
    out = {"fit": "train", "rules": [
        "R-same", "R-edit", "R-ngram", "R-lcs", "majority", "length", "structure",
        "n_cue", "R-pron"],
        "disclosed": ["n_cue@句位", "n_cue@字距桶", "n_cue@m2句位"], "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = run_arm(arm)
    (RES / "battery.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n→ {RES / 'battery.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
