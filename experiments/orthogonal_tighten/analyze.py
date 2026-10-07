#!/usr/bin/env python3
"""P23 判据分析：T0–T5 + 卡−条件化查表（N2 口径，只读 P8 rules.json）→ results/gates.json。

用法：uv run python experiments/orthogonal_tighten/analyze.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P8 = ROOT / "experiments" / "mode_conditioned_skel"
RESULTS = HERE / "results"
L_GROUP = (35, 0, 1, 2)
SIG = ("S1_句式", "S2_nslots", "S3_稀有度b", "S4_合格候选", "S5_首末字", "S6_类型库存")
CUM = tuple(f"A{i + 1}" for i in range(6))
RAND = tuple(f"A{i + 1}r" for i in range(6))
P8_REF = {"F_s42": .156731, "F_s43": .158654, "C_mode_s42": .260577, "C_mode_s43": .267308}


def paired(a: list[int], b: list[int]) -> dict:
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {"delta": round(m, 6), "se": round(se, 6),
            "t": round(m / se, 2) if se > 0 else None,
            "n_pos": sum(1 for x in d if x > 0), "n_neg": sum(1 for x in d if x < 0),
            "n_changed": sum(1 for x in d if x != 0)}


def acc(c: list[int]) -> float:
    return sum(c) / len(c)


def sub(c: list[int], idx: list[int]) -> float:
    return sum(c[i] for i in idx) / len(idx) if idx else float("nan")


def main() -> None:
    ev = json.loads((RESULTS / "eval.json").read_text(encoding="utf-8"))
    rules = json.loads((P8 / "results" / "rules.json").read_text(encoding="utf-8"))
    out: dict = {"n": ev["n"], "cum_order": ev["cum_order"], "seeds": [42, 43],
                 "sizes": ev["sizes"], "main": {}}

    # ---- T0 复现 ----
    r0 = {}
    for seed in (42, 43):
        rec = ev["rows"][f"UP_s{seed}"]["a_bal"]["decode"]
        r0[f"s{seed}"] = {}
        for mine_k, ref_k in (("F", f"F_s{seed}"), ("S1_句式", f"C_mode_s{seed}")):
            m, p = round(rec[mine_k]["acc"], 4), P8_REF[ref_k]
            d = abs(m - p)
            r0[f"s{seed}"][mine_k] = {"mine": m, "p8": p, "abs_diff": round(d, 4),
                                      "pass": bool(d <= 0.0155)}
    out["T0_repro"] = r0

    # ---- 主表 ----
    for split in ("a_bal", "test", "adv2"):
        tab = {}
        for seed in (42, 43):
            rec = ev["rows"][f"UP_s{seed}"][split]
            g, e = rec["gold"], {}
            li = [i for i, x in enumerate(g) if x in L_GROUP]
            ni = [i for i, x in enumerate(g) if x not in L_GROUP]
            e["n"] = len(g)
            e["SE"] = round(math.sqrt(0.25 / len(g)), 4)
            e["L_n"], e["nonL_n"] = len(li), len(ni)
            for d in ("F",) + SIG + CUM + RAND:
                c = rec["decode"][d]["correct"]
                e[d] = {"acc": round(acc(c), 4), "cov": rec["coverage"].get(d),
                        "empty": rec["empty"].get(d),
                        "empty_pct": round(100 * rec["empty"].get(d, 0) / len(g), 1)}
                if li:
                    e[d]["L"] = round(sub(c, li), 4)
                if ni:
                    e[d]["nonL"] = round(sub(c, ni), 4)
            f = rec["decode"]["F"]["correct"]
            cm = rec["decode"]["S1_句式"]["correct"]
            for d in SIG + CUM + RAND:
                e[d]["dF"] = paired(f, rec["decode"][d]["correct"])
                e[d]["dCmode"] = paired(cm, rec["decode"][d]["correct"])
            # 逐累积步的边际（A1 相对 F，A2..A6 相对上一臂）
            prev = f
            e["marginal"] = {}
            for i, d in enumerate(CUM):
                cur = rec["decode"][d]["correct"]
                e["marginal"][d] = paired(prev, cur)
                prev = cur
            e["rand"] = {d: paired(rec["decode"][d]["correct"],
                                   rec["decode"][d + "r"]["correct"])
                         for d in CUM}
            tab[f"s{seed}"] = e
        out["main"][split] = tab

    # ---- T3 卡 − 最强查表 ----
    t3 = {}
    for split in ("a_bal", "test", "adv2"):
        best = rules["splits"][split]["max_naive_disclosure"]
        t3[split] = {"best": best,
                     "per_arm": {d: {f"s{s}": round(out["main"][split][f"s{s}"][d]["acc"] - best, 4)
                                     for s in (42, 43)}
                                 for d in ("F",) + SIG + CUM}}
    out["T3_card_minus_lookup"] = t3

    # ---- T1 累积单调 + 最终臂 ----
    ab = out["main"]["a_bal"]
    t1 = {}
    for seed in (42, 43):
        seq = [ab[f"s{seed}"][d]["acc"] for d in CUM]
        steps = [round(seq[i + 1] - seq[i], 4) for i in range(5)]
        marg = {d: ab[f"s{seed}"]["marginal"][d] for d in CUM}
        t1[f"s{seed}"] = {"seq": seq, "steps": steps,
                          "A6_dCmode": ab[f"s{seed}"]["A6"]["dCmode"],
                          "non_decreasing": all(x >= -1e-9 for x in steps),
                          "marginal": marg}
    t1["pass"] = all(t1[f"s{s}"]["non_decreasing"] for s in (42, 43)) and all(
        t1[f"s{s}"]["A6_dCmode"]["delta"] > 0 and t1[f"s{s}"]["A6_dCmode"]["se"] > 0 and
        t1[f"s{s}"]["A6_dCmode"]["delta"] > 2 * t1[f"s{s}"]["A6_dCmode"]["se"]
        for s in (42, 43))
    out["T1_cumulative"] = t1

    # ---- T2 逐信号单独 + 边际分类 ----
    t2 = {}
    for d in SIG:
        per = {f"s{s}": {"acc": ab[f"s{s}"][d]["acc"],
                         "dF": ab[f"s{s}"][d]["dF"],
                         "dCmode": ab[f"s{s}"][d]["dCmode"]} for s in (42, 43)}
        same = (per["s42"]["dF"]["delta"] > 0) == (per["s43"]["dF"]["delta"] > 0)
        z = (not same) or any(abs(per[f"s{s}"]["dF"]["delta"]) <= per[f"s{s}"]["dF"]["se"]
                              for s in (42, 43))
        both_pos = all(per[f"s{s}"]["dF"]["delta"] > 0 for s in (42, 43))
        weak = any(per[f"s{s}"]["dF"]["delta"] <= 2 * per[f"s{s}"]["dF"]["se"]
                   for s in (42, 43))
        if z:
            call = "≈0（没测出差异）"
        elif both_pos:
            call = "正贡献" + ("（弱：有 seed ≤2SE）" if weak else "（>2SE 同号）")
        else:
            call = "负贡献" + ("（弱：有 seed ≤2SE）" if weak else "（<−2SE 同号）")
        t2[d] = {"standalone": per, "verdict": call}
    # 边际：A1 相对 F，A2..A6 相对上一臂
    marg_cls = {}
    for i, d in enumerate(CUM):
        sig_name = ev["cum_order"][i]
        per = {f"s{s}": ab[f"s{s}"]["marginal"][d] for s in (42, 43)}
        same = (per["s42"]["delta"] > 0) == (per["s43"]["delta"] > 0)
        near0 = (not same) or any(abs(per[f"s{s}"]["delta"]) <= per[f"s{s}"]["se"]
                                  for s in (42, 43))
        both_pos = all(per[f"s{s}"]["delta"] > 0 for s in (42, 43))
        weak = any(per[f"s{s}"]["delta"] <= 2 * per[f"s{s}"]["se"] for s in (42, 43))
        if near0:
            call = "≈0（没测出差异）"
        elif both_pos:
            call = "正贡献" + ("（弱：有 seed ≤2SE）" if weak else "（>2SE 同号）")
        else:
            call = "负贡献" + ("（弱：有 seed ≤2SE）" if weak else "（<−2SE 同号）")
        marg_cls[d] = {"signal": sig_name, "paired": per, "call": call}
    out["T2_per_signal"] = {"standalone": t2, "marginal_in_cum": marg_cls}

    # ---- T4 空集 / L 分列 / 随机对照 ----
    t4 = {}
    for split in ("a_bal", "test", "adv2"):
        t4[split] = {f"s{s}": {d: {"empty": out["main"][split][f"s{s}"][d]["empty"],
                                   "empty_pct": out["main"][split][f"s{s}"][d]["empty_pct"],
                                   "L": out["main"][split][f"s{s}"][d].get("L"),
                                   "nonL": out["main"][split][f"s{s}"][d].get("nonL"),
                                   "rand_acc": out["main"][split][f"s{s}"]
                                   .get(d + "r", {}).get("acc")}
                               for d in CUM} for s in (42, 43)}
    t4["rand_control"] = {d: {f"s{s}": ab[f"s{s}"]["rand"][d] for s in (42, 43)}
                          for d in CUM}
    t4["rand_below"] = all(t4["rand_control"][d][f"s{s}"]["delta"] < 0
                           for s in (42, 43) for d in CUM)
    out["T4"] = t4
    out["T5_sizes"] = ev["sizes"]

    # ---- 判定 ----
    t0_pass = all(r0[f"s{s}"][k]["pass"] for s in (42, 43) for k in ("F", "S1_句式"))
    t3_a6 = t3["a_bal"]["per_arm"]["A6"]
    t3_pass = all(v > 0 for v in t3_a6.values())
    all_pos = all(marg_cls[d]["call"].startswith("正") for d in CUM)
    if t0_pass and t1["pass"] and t3_pass and all_pos:
        verdict = "① 多信号收紧有效且每信号都有贡献"
    elif t0_pass and t1["pass"] and t3_pass:
        verdict = "② 只有点名的部分信号有贡献"
    else:
        verdict = "③ 证据不足（按 PREREG 写死映射：T1/T3 未过）"
    out["gates"] = {"T0": t0_pass, "T1": t1["pass"], "T2": "见 T2_per_signal",
                    "T3_A6_pass": t3_pass, "T4": True, "T5": True}
    out["verdict"] = verdict
    out["old_cards"] = "N/A（未训任何头）"

    (RESULTS / "gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps({
        "T0": r0,
        "T1": {f"s{s}": {"seq": t1[f"s{s}"]["seq"], "steps": t1[f"s{s}"]["steps"],
                        "A6_dCmode": t1[f"s{s}"]["A6_dCmode"]} for s in (42, 43)},
        "T2_marginal": {d: {"sig": v["signal"], "call": v["call"],
                            "d42": v["paired"]["s42"]["delta"],
                            "d43": v["paired"]["s43"]["delta"]} for d, v in marg_cls.items()},
        "T2_standalone": {d: {f"s{s}": t2[d]["standalone"][f"s{s}"]["acc"] for s in (42, 43)}
                          | {"call": t2[d]["verdict"]} for d in SIG},
        "T3_a_bal": {d: v for d, v in t3["a_bal"]["per_arm"].items()},
        "T4_empty_a_bal": {d: {f"s{s}": out["main"]["a_bal"][f"s{s}"][d]["empty_pct"]
                               for s in (42, 43)} for d in CUM},
        "T4_rand_below": t4["rand_below"],
        "T4_rand_delta_main_minus_rand": {d: {f"s{s}": t4["rand_control"][d][f"s{s}"]["delta"]
                                              for s in (42, 43)} for d in CUM},
        "verdict": verdict}, ensure_ascii=False, indent=1))
    print(f"[done] → {RESULTS / 'gates.json'}")


if __name__ == "__main__":
    main()
