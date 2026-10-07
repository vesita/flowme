#!/usr/bin/env python3
"""P22 判据分析：R0–R5 + 卡−条件化查表（N2 口径，只读 P8 rules.json）→ results/gates.json。

用法：uv run python experiments/rarity_routing/analyze.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P8 = ROOT / "experiments" / "mode_conditioned_skel"
RESULTS = HERE / "results"
L_GROUP = (35, 0, 1, 2)
DECODES = ("F", "C_mode", "C_rarity", "C_both", "C_mode_rand", "C_rarity_rand", "C_both_rand")
P8_REF = {"F_s42": .156731, "F_s43": .158654, "C_mode_s42": .260577, "C_mode_s43": .267308}


def paired(a: list[int], b: list[int]) -> dict:
    """配对差 b−a：mean、SE、t。"""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {"delta": round(m, 6), "se": round(se, 6),
            "t": round(m / se, 2) if se > 0 else None,
            "n_pos": sum(1 for x in d if x > 0), "n_neg": sum(1 for x in d if x < 0),
            "n_pairs_changed": sum(1 for x in d if x != 0)}


def acc(corr: list[int]) -> float:
    return sum(corr) / len(corr)


def sub(corr: list[int], keep: list[int]) -> float:
    return sum(corr[i] for i in keep) / len(keep) if keep else float("nan")


def main() -> None:
    ev = json.loads((RESULTS / "eval.json").read_text(encoding="utf-8"))
    rules = json.loads((P8 / "results" / "rules.json").read_text(encoding="utf-8"))
    out: dict = {"n": ev["n"], "struct": ev["struct"], "seeds": [42, 43], "main": {}}

    # ---- R0 复现 ----
    r0 = {}
    for seed in (42, 43):
        rec = ev["rows"][f"UP_s{seed}"]["a_bal"]["decode"]
        r0[f"s{seed}"] = {
            "F": {"mine": round(rec["F"]["acc"], 4), "p8": P8_REF[f"F_s{seed}"]},
            "C_mode": {"mine": round(rec["C_mode"]["acc"], 4), "p8": P8_REF[f"C_mode_s{seed}"]},
        }
        for d in ("F", "C_mode"):
            diff = abs(r0[f"s{seed}"][d]["mine"] - r0[f"s{seed}"][d]["p8"])
            r0[f"s{seed}"][d]["abs_diff"] = round(diff, 4)
            r0[f"s{seed}"][d]["pass"] = bool(diff <= 0.0155)
    out["R0_repro"] = r0

    # ---- 主表（各出口 × 各臂 × 2 seed）+ 配对 ----
    for split in ("a_bal", "test", "adv2"):
        tab = {}
        for seed in (42, 43):
            rec = ev["rows"][f"UP_s{seed}"][split]
            g = rec["gold"]
            li = [i for i, x in enumerate(g) if x in L_GROUP]
            ni = [i for i, x in enumerate(g) if x not in L_GROUP]
            e = {"n": len(g), "SE": round(math.sqrt(0.25 / len(g)), 4),
                 "L_n": len(li), "nonL_n": len(ni)}
            for d in DECODES:
                c = rec["decode"][d]["correct"]
                e[d] = {"acc": round(acc(c), 4)}
                if len(li):
                    e[d]["L"] = round(sub(c, li), 4)
                if len(ni):
                    e[d]["nonL"] = round(sub(c, ni), 4)
            f = rec["decode"]["F"]["correct"]
            for d in ("C_mode", "C_rarity", "C_both", "C_rarity_rand", "C_mode_rand"):
                e[d]["dF"] = paired(f, rec["decode"][d]["correct"])
            e["C_rarity_dCmode"] = paired(rec["decode"]["C_mode"]["correct"],
                                          rec["decode"]["C_rarity"]["correct"])
            e["C_both_dCmode"] = paired(rec["decode"]["C_mode"]["correct"],
                                        rec["decode"]["C_both"]["correct"])
            tab[f"s{seed}"] = e
        out["main"][split] = tab

    # ---- R3 卡 − 条件化查表（P8 口径，只读）----
    r3 = {}
    for split in ("a_bal", "test", "adv2"):
        rr = rules["splits"][split]
        lookup = {"R0_n_slots": rr["n_slots"], "R1_mode_nslots": rr["R1_mode_nslots"],
                  "R2_mode": rr["R2_mode"], "max_naive_complete": rr["max_naive_complete"],
                  "max_naive_disclosure(含最强 n_slots+last_char)":
                      rr["max_naive_disclosure"]}
        best = rr["max_naive_disclosure"]
        e = {"lookup": lookup, "best": best, "per_arm": {}}
        for d in ("F", "C_mode", "C_rarity", "C_both"):
            e["per_arm"][d] = {f"s{seed}": round(
                out["main"][split][f"s{seed}"][d]["acc"] - best, 4) for seed in (42, 43)}
        r3[split] = e
    out["R3_card_minus_lookup"] = r3

    # ---- 机制：a_bal 逐 n_slots 桶 / 逐 gold 骨架 ----
    mech = {}
    for seed in (42, 43):
        rec = ev["rows"][f"UP_s{seed}"]["a_bal"]
        gold, nsl = rec["gold"], rec["ns_lab"]
        by_bucket = defaultdict(lambda: defaultdict(list))
        by_gold = defaultdict(lambda: defaultdict(list))
        for i in range(len(gold)):
            for d in DECODES:
                by_bucket[nsl[i]][d].append(rec["decode"][d]["correct"][i])
                by_gold[gold[i]][d].append(rec["decode"][d]["correct"][i])
        mech[f"s{seed}"] = {
            "by_ns_bucket": {str(k): {d: round(acc(v), 4) for d, v in sorted(dd.items())}
                             for k, dd in sorted(by_bucket.items())},
            "by_ns_bucket_n": {str(k): len(dd["F"]) for k, dd in sorted(by_bucket.items())},
            "by_gold_skel": {str(k): {d: round(acc(v), 4) for d, v in sorted(dd.items())}
                             for k, dd in sorted(by_gold.items())},
        }
    out["mechanism_a_bal"] = mech

    # ---- R4 子集大小分布 ----
    out["R4_subset_sizes"] = ev["struct"]["per_split"]
    out["R4_constant_excluded"] = ev["struct"]["train_unseen_constant_excluded"]
    out["gold_coverage"] = ev["struct"].get("gold_coverage")

    # ---- 判据逐条 ----
    gates = {}
    gates["R0"] = {"pass": all(r0[f"s{s}"][d]["pass"] for s in (42, 43)
                               for d in ("F", "C_mode")), "detail": r0}
    ab = out["main"]["a_bal"]
    r1 = {f"s{s}": ab[f"s{s}"]["C_rarity"]["dF"] for s in (42, 43)}
    gates["R1"] = {"per_seed": r1,
                   "pass": all(v["delta"] > 0 and v["se"] > 0 and
                               v["delta"] > 2 * v["se"] for v in r1.values())}
    r2 = {f"s{s}": ab[f"s{s}"]["C_rarity_dCmode"] for s in (42, 43)}
    r2b = {f"s{s}": ab[f"s{s}"]["C_both_dCmode"] for s in (42, 43)}
    verdict_r2 = "相当" if all(abs(v["delta"]) <= v["se"] for v in r2.values()) else (
        "明显更差" if all(v["delta"] < -2 * v["se"] for v in r2.values()) else "介于两者之间")
    gates["R2"] = {"C_rarity_minus_C_mode": r2, "C_both_minus_C_mode": r2b,
                   "call": verdict_r2}
    best_ab = rules["splits"]["a_bal"]["max_naive_disclosure"]
    r3a = {f"s{s}": round(ab[f"s{s}"]["C_rarity"]["acc"] - best_ab, 4) for s in (42, 43)}
    r3c = {f"s{s}": round(ab[f"s{s}"]["C_mode"]["acc"] - best_ab, 4) for s in (42, 43)}
    gates["R3"] = {"best_free_rule_a_bal": best_ab, "C_rarity_minus_best": r3a,
                   "C_mode_minus_best": r3c, "C_rarity_pass": all(v > 0 for v in r3a.values())}
    sizes = ev["struct"]["per_split"]["a_bal"]
    gates["R4"] = {"C_mode": {"rows": sizes["mode_rows"], "subset": sizes["mode_subset_size"]},
                   "C_rarity": {"rows": sizes["ns_rows"], "subset": sizes["ns_subset_size"]},
                   "fallback": sizes["fallback"],
                   "note": "C-mode 大桶行数 852 / 小桶 91+97；C-rarity 大桶 920 / 小桶 80+40"}
    rand = {f"s{s}": {"C_rarity": ab[f"s{s}"]["C_rarity_rand"]["acc"],
                      "F": ab[f"s{s}"]["F"]["acc"],
                      "C_rarity_main": ab[f"s{s}"]["C_rarity"]["acc"],
                      "dF": ab[f"s{s}"]["C_rarity_rand"]["dF"]} for s in (42, 43)}
    gates["R5"] = {"a_bal_L_n": ab["s42"]["L_n"],
                   "test_groups": {f"s{s}": {"L": out["main"]["test"][f"s{s}"]["C_rarity"].get("L"),
                                             "nonL": out["main"]["test"][f"s{s}"]["C_rarity"].get("nonL"),
                                             "F_L": out["main"]["test"][f"s{s}"]["F"].get("L"),
                                             "F_nonL": out["main"]["test"][f"s{s}"]["F"].get("nonL")}
                                   for s in (42, 43)},
                   "adv2_groups": {f"s{s}": {"L": out["main"]["adv2"][f"s{s}"]["C_rarity"].get("L"),
                                             "nonL": out["main"]["adv2"][f"s{s}"]["C_rarity"].get("nonL"),
                                             "F_L": out["main"]["adv2"][f"s{s}"]["F"].get("L"),
                                             "F_nonL": out["main"]["adv2"][f"s{s}"]["F"].get("nonL")}
                                   for s in (42, 43)},
                   "random_control": rand,
                   "rand_below_main": all(rand[f"s{s}"]["C_rarity"] <
                                          rand[f"s{s}"]["C_rarity_main"] for s in (42, 43))}

    if not gates["R0"]["pass"]:
        verdict = "证据不足（R0 复现未过，先修探针）"
    elif not gates["R1"]["pass"]:
        verdict = "证据不足（R1 未过：C-rarity 在 a_bal 未显著高于 F）"
    elif verdict_r2 in ("相当",):
        verdict = "稀有度路由更通用"
    else:
        verdict = "句式是关键"
    out["gates"] = gates
    out["verdict"] = verdict
    out["old_cards"] = "N/A（本单元未训任何头，PREREG §2 写死）"

    (RESULTS / "gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps({"R0": gates["R0"]["pass"], "R1": gates["R1"],
                      "R2_call": gates["R2"]["call"],
                      "R2_C_rarity": gates["R2"]["C_rarity_minus_C_mode"],
                      "R2_C_both": gates["R2"]["C_both_minus_C_mode"],
                      "R3": gates["R3"], "R4": gates["R4"],
                      "R5_rand": gates["R5"]["random_control"],
                      "verdict": verdict}, ensure_ascii=False, indent=1))
    print(f"[done] → {RESULTS / 'gates.json'}")


if __name__ == "__main__":
    main()
