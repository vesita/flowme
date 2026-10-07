#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0 构念前置门：四条字面规则的上限（只读 data/*.jsonl，纯 CPU，不训练）。

判据照抄 PREREG.md §3/§4：
  四条字面规则 = R-exact(=任务口径 R-same) / R-edit / R-ngram / R-lcs(=R-share 共享字·最长公共子串)
  阈值族在 train 上取 acc 最高者 → test 评测；另报 test 上族内 oracle；**两者取大**为该规则分数。
  门槛：全部 < 1 − 2×SE，test n=1200 ⇒ SE=sqrt(0.25/1200)=1.443pt ⇒ 门槛 97.114%。
  任一 ≥ ⇒ 判「仍是表面可解」并停。
规则只看提及对字面 (s1, s2)，不看上下文（上下文族是 §3 披露项，不在 E0 内）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ---- PREREG §4 常量 ----
N_TEST = 1200
SE = math.sqrt(0.25 / N_TEST)          # 0.0144338 → 1.443pt
GATE = 1 - 2 * SE                      # 0.9711324 → 97.114%

FAMILIES = {
    "R-exact (任务口径 R-same)": ["same"],
    "R-edit": ["edit_abs", "edit_norm"],
    "R-ngram": ["ng1", "ng2"],
    "R-lcs (任务口径 R-share)": ["lcs_ratio", "share_ratio", "contains"],
}


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ngram_jacc(a: str, b: str, n: int) -> float:
    ga = {a[i:i + n] for i in range(len(a) - n + 1)} or {a}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)} or {b}
    return len(ga & gb) / len(ga | gb)


def lcs_sub(a: str, b: str) -> int:
    best = 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best


def metrics(s1: str, s2: str) -> dict:
    m = max(len(s1), len(s2), 1)
    return {
        "same": 1.0 if s1 == s2 else 0.0,
        "edit_abs": float(lev(s1, s2)),
        "edit_norm": lev(s1, s2) / m,
        "ng1": ngram_jacc(s1, s2, 1),
        "ng2": ngram_jacc(s1, s2, 2),
        "lcs_ratio": lcs_sub(s1, s2) / m,
        "share_ratio": len(set(s1) & set(s2)) / m,
        "contains": 1.0 if (s1 in s2 or s2 in s1) else 0.0,
    }


def load(split):
    p = DATA / f"{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def thresholds(vals):
    u = sorted(set(vals))
    ts = [u[0] - 1.0]
    ts += [(u[i] + u[i + 1]) / 2 for i in range(len(u) - 1)]
    ts.append(u[-1] + 1.0)
    return ts


def acc(pred, y):
    return sum(1 for p, t in zip(pred, y) if p == t) / len(y)


def main():
    train, test = load("train"), load("test")
    ytr = [1 if it["label"] == "DIFF" else 0 for it in train]
    yte = [1 if it["label"] == "DIFF" else 0 for it in test]
    mtr = [metrics(it["s1"], it["s2"]) for it in train]
    mte = [metrics(it["s1"], it["s2"]) for it in test]

    rows, all_pass = [], True
    for fam, mnames in FAMILIES.items():
        cands = []
        for mn in mnames:
            tr = [m[mn] for m in mtr]
            te = [m[mn] for m in mte]
            for d in ("le", "ge"):
                for t in thresholds(tr + te):
                    ptr = [1 if ((v <= t) if d == "le" else (v >= t)) else 0 for v in tr]
                    pte = [1 if ((v <= t) if d == "le" else (v >= t)) else 0 for v in te]
                    cands.append({"metric": mn, "dir": d, "t": t,
                                  "acc_tr": acc(ptr, ytr), "acc_te": acc(pte, yte)})
        # train 拟合（平手按候选生成顺序确定性取首个）
        best_tr = max(cands, key=lambda c: c["acc_tr"])
        # test 族内 oracle
        best_te = max(cands, key=lambda c: c["acc_te"])
        score = max(best_tr["acc_te"], best_te["acc_te"])
        passed = score < GATE
        all_pass &= passed
        rows.append({
            "family": fam, "n_candidates": len(cands),
            "train_fit": {"metric": best_tr["metric"], "dir": best_tr["dir"], "t": best_tr["t"],
                          "acc_train": best_tr["acc_tr"], "acc_test": best_tr["acc_te"]},
            "test_oracle": {"metric": best_te["metric"], "dir": best_te["dir"], "t": best_te["t"],
                            "acc_test": best_te["acc_te"]},
            "score": score, "se": SE, "gate": GATE, "below_gate": passed,
        })
        print(f"{fam}: n_cand={len(cands)} "
              f"train拟合({best_tr['metric']},{best_tr['dir']},t={best_tr['t']:.4f}) "
              f"train_acc={best_tr['acc_tr']:.4%} → test_acc={best_tr['acc_te']:.4%} | "
              f"test oracle({best_te['metric']},{best_te['dir']},t={best_te['t']:.4f})="
              f"{best_te['acc_te']:.4%} | 取大={score:.4%} "
              f"门槛={GATE:.4%} {'<通过>' if passed else '≥未过(表面可解)'}")

    verdict = "构念通过（可继续建卡）" if all_pass else "仍是表面可解（停）"
    out = {"n_train": len(train), "n_test": len(test), "se": SE, "gate": GATE,
           "rules": rows, "all_below_gate": all_pass, "verdict": verdict}
    (HERE / "e0_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"E0 verdict: {verdict} (all_below_gate={all_pass})")
    raise SystemExit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
