#!/usr/bin/env python3
"""P8 判据汇总：N0–N6 逐条实测 + 配对 SE + 门禁 → results/gates.json。

用法：uv run python experiments/mode_conditioned_skel/analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modes as MM  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ("A", "U", "P", "UP")
SEEDS = (42, 43)
SPLITS = ("test", "adv2", "a_bal")
L_GROUP = (35, 0, 1, 2)
# 既有数字（N0 对账；来源 = skeleton_leak/REPORT.md §3 表 + bag_modules 复评）
REF = {
    ("A", "test"): (0.6396, 0.6424), ("A", "adv2"): (0.2310, 0.2390),
    ("A", "a_bal"): (0.0635, 0.0663), ("U", "test"): (0.8328, 0.8340),
    ("U", "adv2"): (0.3490, 0.3540), ("U", "a_bal"): (0.1308, 0.1298),
    ("P", "test"): (0.7708, 0.7764), ("P", "adv2"): (0.3160, 0.3100),
    ("P", "a_bal"): (0.0837, 0.0846), ("UP", "test"): (0.8416, 0.8440),
    ("UP", "adv2"): (0.3800, 0.3760), ("UP", "a_bal"): (0.1567, 0.1587),
}
NOISE_BAND = {"pronoun": 0.0283, "sentiment": 0.0041,
              "relation": 0.0139, "person": 0.0033}


def se(n: int) -> float:
    return math.sqrt(0.25 / n)


def paired(a: list[int], b: list[int]) -> dict:
    """Δ = b − a 的逐行配对统计。"""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n,
            "gt_2se": bool(mu > 2 * s)}


def main() -> None:
    ev = json.loads((RESULTS / "eval.json").read_text(encoding="utf-8"))
    ru = json.loads((RESULTS / "rules.json").read_text(encoding="utf-8"))
    mp = json.loads((RESULTS / "mapping.json").read_text(encoding="utf-8"))
    pr = json.loads((RESULTS / "probe.json").read_text(encoding="utf-8"))

    out: dict = {"n0": {}, "cells": {}, "paired": {}, "n2": {}, "n4": {}, "n5": {},
                 "fallback": ev["fallback_empty_subset"]}

    # ---------------- N0：F 复现既有数字 ----------------
    n0_ok = True
    for arm in ARMS:
        for sp in SPLITS:
            n = ev["n"][sp]
            s = se(n)
            for i, sd in enumerate(SEEDS):
                f = ev["rows"][f"{arm}_s{sd}"][sp]["F"]["acc"]
                ref = REF[(arm, sp)][i]
                d = round(f - ref, 6)
                ok = abs(d) <= s
                n0_ok = n0_ok and ok
                out["n0"][f"{arm}_s{sd}|{sp}"] = {"mine": f, "ref": ref, "delta": d,
                                                  "SE": round(s, 6), "pass": bool(ok)}
    out["n0_pass"] = bool(n0_ok)
    print(f"[N0] F 复现既有数字 = {n0_ok}（12 组 × 2 seed，阈 = 各出口 SE）", flush=True)

    # ---------------- 逐格 acc + 配对 Δ ----------------
    for sp in SPLITS:
        n = ev["n"][sp]
        gold = ev["rows"]["UP_s42"][sp]["gold"]
        idx = {"all": list(range(n)),
               "L": [i for i, g in enumerate(gold) if g in L_GROUP],
               "nonL": [i for i, g in enumerate(gold) if g not in L_GROUP]}
        for grp, ii in idx.items():
            if not ii:
                continue
            for arm in ARMS:
                for sd in SEEDS:
                    rec = ev["rows"][f"{arm}_s{sd}"][sp]
                    key = f"{arm}_s{sd}|{sp}|{grp}"
                    cell = {"n": len(ii), "SE": round(se(len(ii)), 6)}
                    for d in ("F", "C_rule", "C_pred", "C_predML", "C_soft", "C_softlr",
                              "C_rule_B", "C_pred_B", "C_rule_rand", "C_pred_rand", "C_perm"):
                        c = rec[d]["correct"]
                        cell[d] = round(sum(c[i] for i in ii) / len(ii), 6)
                    out["cells"][key] = cell
            # 配对 Δ（只报 UP 与 A，避免表爆炸；其余臂在 cells 里可算）
            for arm in ARMS:
                for sd in SEEDS:
                    rec = ev["rows"][f"{arm}_s{sd}"][sp]
                    f_ = [rec["F"]["correct"][i] for i in ii]
                    pk = {}
                    for d in ("C_rule", "C_pred", "C_predML", "C_soft", "C_softlr",
                              "C_pred_B", "C_rule_rand", "C_pred_rand", "C_perm"):
                        pk[d] = paired(f_, [rec[d]["correct"][i] for i in ii])
                    out["paired"].setdefault(f"{arm}_s{sd}|{sp}|{grp}", pk)

    # ---------------- N2：卡 − 条件化免费规则 ----------------
    for sp in SPLITS:
        rr = ru["splits"][sp]
        for grp in ("all", "L", "nonL"):
            if grp == "all":
                r1, mxp, mxc = rr["R1_mode_nslots"], rr["max_naive_plus"], rr["max_naive_complete"]
                r2, r4 = rr["R2_mode"], rr["R4_mode_Bsubset"]
                r3 = rr["R3_oracle_mode_nslots"]
                r0 = rr["n_slots"]
                nn = rr["n"]
            else:
                g = rr["groups"][grp]
                if g["n"] == 0:
                    continue
                r1, mxp, mxc = g["R1_mode_nslots"], g["max_naive_plus"], g["max_naive_complete"]
                r2, r4 = g["R2_mode"], g["R4_mode_Bsubset"]
                r3 = g["R3_oracle_mode_nslots"]
                r0 = g["n_slots"]
                nn = g["n"]
            key = f"{sp}|{grp}"
            if grp == "all":
                disc, maxd = rr.get("disclosure_combo_lookup"), rr.get("max_naive_disclosure")
            else:
                g_ = rr["groups"][grp]
                disc, maxd = g_.get("disclosure_combo_lookup"), g_.get("max_naive_disclosure")
            rec = {"n": nn, "SE": round(se(nn), 6), "R0_n_slots": r0,
                   "R1_mode_nslots": r1, "R2_mode": r2, "R3_oracle": r3, "R4_B": r4,
                   "max_naive_complete": mxc, "max_naive_plus": mxp,
                   "disclosure_combo": disc, "max_naive_disclosure": maxd,
                   "model": {}}
            for arm in ARMS:
                for sd in SEEDS:
                    ck = f"{arm}_s{sd}|{sp}|{grp}"
                    if ck not in out["cells"]:
                        continue
                    c = out["cells"][ck]
                    rec["model"][f"{arm}_s{sd}"] = {
                        "F": c["F"], "C_pred": c["C_pred"],
                        "C_pred_minus_R1": round(c["C_pred"] - r1, 6),
                        "C_rule_minus_R1": round(c["C_rule"] - r1, 6),
                        "C_pred_minus_maxplus": round(c["C_pred"] - mxp, 6),
                        "C_pred_minus_maxcomplete": round(c["C_pred"] - mxc, 6),
                        "C_pred_minus_maxdisclosure": (round(c["C_pred"] - maxd, 6)
                                                       if maxd else None),
                        "delta_model_CminusF": round(c["C_pred"] - c["F"], 6),
                        "delta_rule_R1minusR0": round(r1 - r0, 6)}
            out["n2"][key] = rec

    # ---------------- N4：句式分类器 ----------------
    tab = MM.skeleton_table()
    for sp in SPLITS:
        rec = ev["rows"]["UP_s42"][sp]
        gm, rm = rec["gold_mode"], rec["rule_mode"]
        k = ["陈述", "疑问", "祈使", "感叹", "反问"]
        ci = {m: i for i, m in enumerate(k)}
        conf = [[0] * 5 for _ in range(5)]
        for a, b in zip(gm, rm):
            conf[ci[a]][ci[b]] += 1
        acc = sum(1 for a, b in zip(gm, rm) if a == b) / len(gm)
        # 3 类（陈述/疑问/祈使）macro-F1
        f1s = []
        for c in range(3):
            tp = conf[c][c]
            fp = sum(conf[r][c] for r in range(5) if r != c)
            fn = sum(conf[c][r] for r in range(5) if r != c)
            p_ = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p_ * rc / (p_ + rc) if p_ + rc else 0.0)
        out["n4"][f"rule_g|{sp}"] = {"n": len(gm), "acc": round(acc, 4),
                                     "macro_f1_3": round(sum(f1s) / 3, 4),
                                     "recall_3": [round(conf[c][c] / max(1, sum(conf[c])), 4)
                                                  for c in range(3)],
                                     "confusion_gold_rows_pred_cols": conf}
    out["n4"]["probe"] = {sd: {s: {kk: r["splits"][s][kk] for kk in
                                   ("acc", "macro_f1_3cls", "recall", "confusion")}
                               for s in SPLITS}
                          for sd, r in pr["seeds"].items()}
    out["n4"]["probe_val"] = {sd: {"val_acc": r["val_acc"], "best_epoch": r["best_epoch"],
                                   "epochs_run": r["epochs_run"]} for sd, r in pr["seeds"].items()}
    # 探针预测的类别边际（是否退化为多数类）
    for sd, r in pr["seeds"].items():
        for s in SPLITS:
            am = r["pred"][s]["argmax"]
            cnt = Counter(MM.MODES[i] for i in am)
            out["n4"].setdefault("probe_pred_marginal", {})[f"s{sd}|{s}"] = {
                "n": len(am), "dist": dict(cnt)}

    # ---------------- N5：随机标签对照 + 老卡 ----------------
    for sp in SPLITS:
        for grp in ("all",):
            for arm in ARMS:
                for sd in SEEDS:
                    ck = f"{arm}_s{sd}|{sp}|{grp}"
                    pk = out["paired"].get(f"{arm}_s{sd}|{sp}|{grp}")
                    if not pk:
                        continue
                    cell = out["cells"][ck]
                    thr_maj = None
                    out["n5"].setdefault(f"{arm}_s{sd}|{sp}", {
                        "F": cell["F"], "C_rule": cell["C_rule"],
                        "C_rule_rand": cell["C_rule_rand"],
                        "C_pred_rand": cell["C_pred_rand"],
                        "C_perm": cell["C_perm"],
                        "d_rand_gold_vs_F": pk["C_rule_rand"],
                        "d_rand_rule_vs_F": pk["C_pred_rand"],
                        "d_perm_vs_F": pk["C_perm"],
                        "d_Crule_vs_F": pk["C_rule"],
                        "Crule_minus_Crand": round(cell["C_rule"] - cell["C_rule_rand"], 6),
                        "rand_le_F_plus_2se": bool(pk["C_rule_rand"]["delta"]
                                                   <= 2 * pk["C_rule_rand"]["se"]),
                        "rand_lt_Crule": bool(cell["C_rule_rand"] < cell["C_rule"]),
                        "rand_le_majority_2se": thr_maj})

    # 多数类阈值（a_bal 用既有 .0385+.0308；test/adv2 用 rules 里的 majority_own+2SE）
    for sp in SPLITS:
        rr = ru["splits"][sp]
        n = rr["n"]
        thr = round(rr["majority_own"] + 2 * se(n), 4)
        for arm in ARMS:
            for sd in SEEDS:
                k = f"{arm}_s{sd}|{sp}"
                if k in out["n5"]:
                    out["n5"][k]["majority_own"] = rr["majority_own"]
                    out["n5"][k]["thr_majority_2se"] = thr
                    out["n5"][k]["rand_le_majority_2se"] = bool(
                        out["n5"][k]["C_rule_rand"] <= thr)

    before = (RESULTS / "sha256_before.txt").read_text(encoding="utf-8")
    out["n5"]["old_cards"] = {"sha256_before": before,
                              "noise_band_pt": NOISE_BAND,
                              "note": "本单元不写任何 checkpoint；Δ = after − before 由哈希与权重只读构造保证"}

    # ---------------- 机制分解：按 gold 句式分组的 acc（UP 两 seed）----------------
    out["mech"] = {}
    for sp in SPLITS:
        for sd in SEEDS:
            rec = ev["rows"][f"UP_s{sd}"][sp]
            gm, rm = rec["gold_mode"], rec["rule_mode"]
            modes = sorted(set(gm))
            rec2 = {}
            for m in modes:
                ii = [i for i, x in enumerate(gm) if x == m]
                if not ii:
                    continue
                rec2[m] = {"n": len(ii),
                           "SE": round(se(len(ii)), 4),
                           **{d: round(sum(rec[d]["correct"][i] for i in ii) / len(ii), 4)
                              for d in ("F", "C_rule", "C_pred", "C_predML", "C_perm",
                                        "C_rule_rand")}}
                rec2[m]["贡献_Cpred减F"] = round(
                    (sum(rec["C_pred"]["correct"][i] for i in ii)
                     - sum(rec["F"]["correct"][i] for i in ii)) / ev["n"][sp], 4)
            out["mech"].setdefault(f"UP_s{sd}|{sp}", rec2)

    # 按 gold 句式分组的规则错分（谁被掩错）
    out["mode_error_by_gold"] = {}
    for sp in SPLITS:
        rec = ev["rows"]["UP_s42"][sp]
        gm, rm = rec["gold_mode"], rec["rule_mode"]
        tb: dict = Counter()
        for a, b in zip(gm, rm):
            tb[f"{a}->{b}"] += 1
        out["mode_error_by_gold"][sp] = dict(tb)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"[done] → {RESULTS / 'gates.json'}", flush=True)


if __name__ == "__main__":
    main()
