#!/usr/bin/env python3
"""P21 结果汇总：主表 + M0–M5 逐条 + 配对 SE（臂间、真 vs 随机标签）。"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ARMS = ("M", "MP", "MP+")
SEEDS = (42, 43)


def load(arm, seed, rand=False) -> dict:
    n = f"eval_{arm}_s{seed}{'_rand' if rand else ''}.json"
    return json.loads((RES / n).read_text(encoding="utf-8"))


def paired(a: list, b: list) -> dict:
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(se, 6),
            "t": round(mu / se, 3) if se else None}


def main() -> None:
    D = {(a, s, r): load(a, s, r) for a in ARMS for s in SEEDS for r in (False, True)}
    out: dict = {}

    # ---- M0 机会（实测）----
    ref = D[("M", 42, False)]
    ch = ref["chance"]
    out["M0"] = {"机会_R_mc": ch["uniform_random_R"], "机会_P_perm": ch["permutation_P"],
                 "盲预测器": ch["permutation_P"]["blind_const"],
                 "门槛": ch["threshold"],
                 "机会_R 各臂一致性": sorted({D[k]["chance"]["uniform_random_R"]["mean"]
                                              for k in D})}
    gate40 = 0.5 + 2 * math.sqrt(0.25 / 280)
    out["gate40"] = round(gate40, 6)

    # ---- 主表 ----
    tbl = []
    for a in ARMS:
        for s in SEEDS:
            for r in (False, True):
                d = D[(a, s, r)]
                h = d["heldout"]["b_all"]
                b1 = d["heldout"]["b1_word"]
                b2 = d["heldout"]["b2_order"]
                tbl.append({"arm": a, "seed": s, "rand": r,
                            "2AFC": h["twoafc"], "2AFC_b1": b1["twoafc"],
                            "2AFC_b2": b2["twoafc"],
                            "p40": h["pair_success_40"], "p40_b1": b1["pair_success_40"],
                            "per_side": h["per_side_acc"],
                            "gate2AFC": d["chance"]["threshold"],
                            "full560_2AFC": d["full560_insample"]["twoafc"],
                            "full560_p40": d["full560_insample"]["pair_success_40"],
                            "M1": d["chance"]["M1_pass"]})
    out["main"] = tbl

    # 两 seed SE（把 2 seed 当两次重复报均值 ± 极差；逐对 SE 另列）
    out["two_seed"] = {}
    for a in ARMS:
        v = [D[(a, s, False)]["heldout"]["b_all"]["twoafc"] for s in SEEDS]
        p = [D[(a, s, False)]["heldout"]["b_all"]["pair_success_40"] for s in SEEDS]
        out["two_seed"][a] = {"2AFC": v, "2AFC_mean": round(sum(v) / 2, 6),
                              "p40": p, "p40_mean": round(sum(p) / 2, 6),
                              "se_metric_280": round(math.sqrt(0.25 / 280), 6)}

    # ---- M1 / M2 ----
    out["M1"] = {a: {"s42": D[(a, 42, False)]["chance"]["M1_pass"],
                     "s43": D[(a, 43, False)]["chance"]["M1_pass"]} for a in ARMS}
    out["M2"] = {a: {"s42_p40": D[(a, 42, False)]["heldout"]["b_all"]["pair_success_40"],
                     "s43_p40": D[(a, 43, False)]["heldout"]["b_all"]["pair_success_40"],
                     "gate": round(gate40, 6),
                     "pass": [D[(a, s, False)]["heldout"]["b_all"]["pair_success_40"] > gate40
                              for s in SEEDS]} for a in ARMS}

    # ---- 臂间配对 SE（同一批 held-out 对逐 0/1 之差）----
    out["arm_vs_arm"] = {}
    for a in ARMS:
        for b in ARMS:
            if a >= b:
                continue
            key = f"{a}-{b}"
            out["arm_vs_arm"][key] = {}
            for s in SEEDS:
                x = D[(a, s, False)]["heldout"]["b_all"]
                y = D[(b, s, False)]["heldout"]["b_all"]
                out["arm_vs_arm"][key][f"s{s}_2AFC"] = paired(x["twoafc_list"], y["twoafc_list"])
                out["arm_vs_arm"][key][f"s{s}_p40"] = paired(x["pair40_list"], y["pair40_list"])

    # ---- M3 地板 ----
    fl = ref["floor"]
    out["M3"] = {"by_rule": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_list")}
                             for k, v in fl["by_rule"].items()},
                 "max_naive_complete": fl["max_naive_complete"],
                 "max_naive_complete_rule": fl["max_naive_complete_rule"],
                 "R_chain_ref": {k: v for k, v in fl["R_chain_ref"].items()
                                 if not k.endswith("_list")},
                 "max_pair_success_rule": fl["max_pair_success_rule"],
                 "max_twoafc_rule": fl["max_twoafc_rule"],
                 "L_split": fl["L_split"],
                 "机会(实测)": {"R": ch["uniform_random_R"]["mean"],
                               "P": ch["permutation_P"]["mean"]}}

    # ---- M4 随机标签（配对 SE：真臂 vs rand 臂）----
    out["M4"] = {}
    for a in ARMS:
        out["M4"][a] = {}
        for s in SEEDS:
            x = D[(a, s, False)]["heldout"]["b_all"]
            y = D[(a, s, True)]["heldout"]["b_all"]
            out["M4"][a][f"s{s}"] = {
                "true_2AFC": x["twoafc"], "rand_2AFC": y["twoafc"],
                "true_p40": x["pair_success_40"], "rand_p40": y["pair_success_40"],
                "paired_true_minus_rand": paired(x["twoafc_list"], y["twoafc_list"]),
                "rand_below_chance_gate": bool(
                    y["twoafc"] <= ch["uniform_random_R"]["mean"]
                    - 2 * math.sqrt(0.25 / 280))}

    # ---- M5 机制 ----
    out["M5"] = {}
    for a in ARMS:
        for s in SEEDS:
            d = D[(a, s, False)]
            sc = d["probe"]["shuffle_content"]
            out["M5"][f"{a}_s{s}"] = {
                "shuf_content_pairs": f"{sc['n_survivor_pairs']}/{sc['n_total_pairs']}",
                "true_2AFC": sc.get("true_on_survivors", {}).get("twoafc"),
                "shuf_2AFC": sc.get("shuffled", {}).get("twoafc"),
                "true_p40": sc.get("true_on_survivors", {}).get("pair_success_40"),
                "shuf_p40": sc.get("shuffled", {}).get("pair_success_40"),
                "shuffle_all_2AFC": d["probe"]["shuffle_all"]["twoafc"],
                "shuffle_all_p40": d["probe"]["shuffle_all"]["pair_success_40"],
                "true_rel_2AFC": d["probe"]["shuffle_all_true"]["twoafc"],
                "aux_only": d["probe"]["aux_only"]["applicable"]}

    # ---- 对账（in-sample 全量 560，P4 数字）----
    out["P4_ref"] = {"M": [0.0, 0.0], "MP": [0.0, 0.0], "MP+": [0.0192, 0.0173],
                     "gate_520": 0.543853, "P4_2AFC_MP": [0.621154, 0.609615],
                     "P4_2AFC_MP+": [0.540385, 0.540385]}

    (HERE / "results" / "tables.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
