#!/usr/bin/env python3
"""P20 判据汇总：H0–H5 逐条 + 配对 Δ/SE + 探索性（跑后）分解。"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "sentence_mode"))
import labels as L  # noqa: E402

RESULTS = HERE / "results"
ARMS = ("A", "B", "C")
SEEDS = (42, 43)


def paired(d: list[int]) -> dict:
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return {"delta": round(m, 6), "se": round(se, 6),
            "t": round(m / se, 3) if se > 0 else None,
            "pass_2se": bool(abs(m) > 2 * se), "same_sign_required": True, "n": n}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def detail(name: str) -> dict:
    return json.loads((RESULTS / f"detail_{name}.json").read_text(encoding="utf-8"))["detail"]


def main() -> None:
    runs = {(a, s): load(f"{a}_s{s}") for a in ARMS for s in SEEDS}
    dets = {(a, s): detail(f"{a}_s{s}") for a in ARMS for s in SEEDS}
    floor = json.loads((RESULTS / "floor.json").read_text(encoding="utf-8"))
    probes = {(a, s): json.loads((RESULTS / f"probe_{a}_s{s}.json").read_text(encoding="utf-8"))
              for a in ARMS for s in SEEDS}

    out: dict = {"H0": {}, "H1": {}, "H2": {}, "H3": {}, "H4": {}, "H5": {}, "M": {},
                 "exploratory": {}}

    # ---------------- H0 ----------------
    for a in ARMS:
        r = runs[(a, 42)]
        out["H0"][a] = {"params": r["params"]["total"], "per_group": r["params"]["per_group"],
                        "step_ms": r["step_ms"], "wall_sec": r["wall_sec"],
                        "train_wall_sec": r["train_wall_sec"],
                        "peak_mem_GB": round(r["peak_mem_bytes"] / 1e9, 3),
                        "steps": r["steps"], "batch": r["batch_total"]}
    base = out["H0"]["A"]["params"]
    for a in ARMS:
        out["H0"][a]["params_vs_A_pct"] = round(
            100 * (out["H0"][a]["params"] - base) / base, 2)

    # ---------------- H1 / H2 / H5（配对逐行 0/1）----------------
    # 注：train split 只存了标量（未存逐行 0/1）⇒ H5 的表内项只报绝对值，见 H5_abs
    for key, tag in (("S_test_mask", "H1"), ("T_test", "H2"), ("S_test_keep", "H5")):
        for b in ("B", "C"):
            for s in SEEDS:
                ca = dets[("A", s)][f"{key}_correct"]
                cb = dets[(b, s)][f"{key}_correct"]
                diff = [int(y) - int(x) for x, y in zip(ca, cb)]
                out[tag].setdefault(f"{key}|{b}-A", {})[f"s{s}"] = paired(diff)
            # 2 seed 同号
            signs = {out[tag][f"{key}|{b}-A"][f"s{s}"]["delta"] > 0 for s in SEEDS}
            rec = out[tag][f"{key}|{b}-A"]
            rec["same_sign_2seeds"] = len(signs) == 1
            rec["both_pass_2se"] = all(rec[f"s{s}"]["pass_2se"] for s in SEEDS)
            rec["H_pass"] = bool(rec["both_pass_2se"] and rec["same_sign_2seeds"]
                                 and all(rec[f"s{s}"]["delta"] > 0 for s in SEEDS))

    # H1 需要 **正** 增益；H2 需要 B/C ≥ A − 2SE ⇒ delta ≥ −2SE
    for b in ("B", "C"):
        k = f"S_test_mask|{b}-A"
        out["H1"]["H_pass"] = out["H1"].get("H_pass", False) or out["H1"][k]["H_pass"]
        for s in SEEDS:
            r = out["H2"][f"T_test|{b}-A"][f"s{s}"]
            r["pass_H2"] = bool(r["delta"] >= -2 * r["se"])
        out["H2"][f"{b}_no_degradation"] = all(
            out["H2"][f"T_test|{b}-A"][f"s{s}"]["pass_H2"] for s in SEEDS)

    # ---------------- H3 地板 ----------------
    out["H3"] = {
        "S_mask": {k: floor["S"]["exits"]["mask"][k] for k in
                   ("n", "max_naive", "max_naive_rule", "majority", "majority_plus_2se",
                    "one_minus_2se", "usable_rule_cannot_solve")},
        "S_keep": {k: floor["S"]["exits"]["keep"][k] for k in
                   ("n", "max_naive", "max_naive_rule", "majority", "one_minus_2se",
                    "usable_rule_cannot_solve")},
        "T": {k: floor["T"][k] for k in
              ("n_test", "max_naive", "max_naive_rule", "majority", "one_minus_2se",
               "one_minus_2se_at_floor", "usable_rule_cannot_solve")},
        "model_vs_floor_S_mask": {f"{a}_s{s}": round(
            runs[(a, s)]["eval"]["S_test_mask"]["acc"] - floor["S"]["exits"]["mask"]["max_naive"], 6)
            for a in ARMS for s in SEEDS},
        "model_vs_floor_T": {f"{a}_s{s}": round(
            runs[(a, s)]["eval"]["T_test"]["acc"] - floor["T"]["max_naive"], 6)
            for a in ARMS for s in SEEDS},
    }

    # ---------------- H4 ----------------
    for probe_name in ("skel", "trig_pos", "ent", "b_pairs", "c_pairs"):
        out["H4"][probe_name] = {f"{a}_s{s}": probes[(a, s)]["probes"][probe_name]
                                 for a in ARMS for s in SEEDS}
    # ---------------- H5 / M 绝对值 ----------------
    out["H5_abs"] = {f"{a}_s{s}": {k: runs[(a, s)]["eval"][k]["acc"] for k in
                                   ("S_test_keep", "S_train_mask", "T_train", "S_test_mask")}
                     for a in ARMS for s in SEEDS}
    out["M"] = {f"{a}_s{s}": runs[(a, s)]["eval"]["M_shuffle"] for a in ARMS for s in SEEDS}

    # ---------------- 探索性（跑后，非预注册）----------------
    # 在免费规则 R6 预测错的行上重算 acc（"规则解决不了的子集"）
    import data as D
    te = D.load_sentence_mode()["test"]
    r6 = [L.MODE_ID[L.label_defn_on(r["exit"]["mask"])] for r in te]
    gold = [r["mode"] for r in te]
    fail = [i for i, (p, g) in enumerate(zip(r6, gold)) if p != g]
    out["exploratory"]["rule_fail_subset"] = {
        "n": len(fail), "pct": round(len(fail) / len(gold), 4),
        "acc": {f"{a}_s{s}": round(sum(dets[(a, s)]["S_test_mask_correct"][i]
                                       for i in fail) / len(fail), 6)
                for a in ARMS for s in SEEDS}}
    pass_idx = [i for i in range(len(gold)) if i not in set(fail)]
    out["exploratory"]["rule_pass_subset"] = {
        "n": len(pass_idx),
        "acc": {f"{a}_s{s}": round(sum(dets[(a, s)]["S_test_mask_correct"][i]
                                       for i in pass_idx) / len(pass_idx), 6)
                for a in ARMS for s in SEEDS}}
    # 类别分解
    per_class: dict = {}
    for cls in sorted(set(gold)):
        idx = [i for i, g in enumerate(gold) if g == cls]
        per_class[str(cls)] = {"n": len(idx),
                               "mode_name": te[idx[0]]["mode_name"],
                               **{f"{a}_s{s}": round(
                                   sum(dets[(a, s)]["S_test_mask_correct"][i] for i in idx)
                                   / len(idx), 4) for a in ARMS for s in SEEDS}}
    out["exploratory"]["per_class"] = per_class

    (RESULTS / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("H0", "H1", "H2")},
                     ensure_ascii=False, indent=2))
    print("[done] results/summary.json")


if __name__ == "__main__":
    main()
