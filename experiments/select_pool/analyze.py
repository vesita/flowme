#!/usr/bin/env python3
"""select_pool 结果汇总：按 PREREG §4 逐条判 Q1–Q5 并给三选一判定。

用法：uv run python experiments/select_pool/analyze.py
     → 打印 Markdown 表 + 写 results/summary.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ("A", "B", "C", "D", "Aplus")
SEEDS = (42, 43)

SE1250 = math.sqrt(0.25 / 1250)            # 0.0141421 → 1.414pt
TWO_SE = 2 * SE1250 * 100                   # 2.828pt（百分点）
# Q4：B 族 select_rerank/REPORT.md §3.1 的 heldout_pair
B族 = {"s42": 49.52, "s43": 51.04}
B族_SHIFT = {"s42": 92.56, "s43": 92.48}


def load(arm: str, seed: int, rand: bool = False) -> dict | None:
    n = f"{arm}_s{seed}{'_rand' if rand else ''}.json"
    p = RESULTS / n
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float) -> float:
    return round(x * 100, 4)


def main() -> None:
    out: dict = {"se_pt": round(SE1250 * 100, 4), "two_se_pt": round(TWO_SE, 4),
                 "arms": {}, "Q": {}, "missing": []}

    # ---- 主表 -------------------------------------------------------------
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            r = load(arm, seed)
            if r is None:
                out["missing"].append(f"{arm}_s{seed}")
                continue
            h = r["eval"]["adv_by_ctype"]["heldout_pair"]
            s = r["eval"]["adv_by_ctype"]["shifted_pos"]
            rows.append({"arm": arm, "seed": seed,
                         "heldout": pct(h["acc"]), "shifted": pct(s["acc"]),
                         "adv": pct(r["eval"]["adv"]["acc"]),
                         "test": pct(r["eval"]["test"]["acc"]),
                         "train": pct(r["eval"]["train"]["acc"]),
                         "trainable": r["freeze"]["trainable_params"],
                         "agg_params": r["freeze"]["agg_params"],
                         "head_params": r["freeze"]["head_params"],
                         "encoder_trainable": r["freeze"]["encoder_trainable"],
                         "parity": r.get("parity"), "wall": r["wall_sec"]})
            out["arms"].setdefault(arm, {})[f"s{seed}"] = rows[-1]

    # ---- Q4：臂 A 复现 B 族 ----------------------------------------------
    q4 = {"b族": {f"heldout_{k}": v for k, v in B族.items()},
          "arm_A": {}, "pass": True}
    for seed in SEEDS:
        r = load("A", seed)
        if r is None:
            q4["pass"] = False
            continue
        h = pct(r["eval"]["adv_by_ctype"]["heldout_pair"]["acc"])
        sh = pct(r["eval"]["adv_by_ctype"]["shifted_pos"]["acc"])
        d = h - B族[f"s{seed}"]
        ds = sh - B族_SHIFT[f"s{seed}"]
        q4["arm_A"][f"s{seed}"] = {"heldout": h, "delta_vs_b族": round(d, 4),
                                   "shifted": sh, "delta_shifted": round(ds, 4),
                                   "within_2se": abs(d) <= TWO_SE}
        if abs(d) > TWO_SE:
            q4["pass"] = False
        q4["q4a_parity"] = r.get("parity")
    out["Q"]["Q4"] = q4

    # ---- Q1：某臂 heldout > 臂 A，余量 > 2×SE，两 seed 同号 ----------------
    base = {}
    for seed in SEEDS:
        r = load("A", seed)
        if r:
            base[seed] = pct(r["eval"]["adv_by_ctype"]["heldout_pair"]["acc"])
    q1 = {"base_A": base, "two_se_pt": round(TWO_SE, 4), "per_arm": {}, "passing": []}
    for arm in ARMS:
        if arm == "A":
            continue
        arm_res = {}
        ok = True
        for seed in SEEDS:
            r, a = load(arm, seed), load("A", seed)
            if r is None or a is None or seed not in base:
                arm_res[f"s{seed}"] = "缺跑"
                ok = False
                continue
            h = pct(r["eval"]["adv_by_ctype"]["heldout_pair"]["acc"])
            d = h - base[seed]
            arm_res[f"s{seed}"] = {"heldout": h, "delta_vs_A": round(d, 4),
                                   "delta_over_se": round(d / (SE1250 * 100), 2),
                                   "over_2se": d > TWO_SE}
            if not (d > TWO_SE):
                ok = False
        arm_res["Q1_pass"] = ok
        q1["per_arm"][arm] = arm_res
        if ok:
            q1["passing"].append(arm)
    out["Q"]["Q1"] = q1

    # ---- Q2：过 Q1 的臂 shifted 不显著低于 A -------------------------------
    q2 = {"per_arm": {}, "pass": True}
    for arm in q1["passing"]:
        ok = True
        per = {}
        for seed in SEEDS:
            r, a = load(arm, seed), load("A", seed)
            if r is None or a is None:
                ok = False
                continue
            s_ = pct(r["eval"]["adv_by_ctype"]["shifted_pos"]["acc"])
            sa = pct(a["eval"]["adv_by_ctype"]["shifted_pos"]["acc"])
            d = s_ - sa
            per[f"s{seed}"] = {"shifted": s_, "A": sa, "delta": round(d, 4),
                               "pass": d >= -TWO_SE}
            if d < -TWO_SE:
                ok = False
        per["Q2_pass"] = ok
        q2["per_arm"][arm] = per
        if not ok:
            q2["pass"] = False
    if not q1["passing"]:
        q2["pass"] = None          # 没有臂过 Q1 ⇒ Q2 无对象（不判）
    out["Q"]["Q2"] = q2

    # ---- Q3：随机标签掉回 50% ---------------------------------------------
    limit = 50 + TWO_SE
    q3 = {"limit_pt": round(limit, 4), "runs": {}, "pass": True}
    for arm in ARMS:
        for seed in SEEDS:
            r = load(arm, seed, rand=True)
            if r is None:
                if seed == 42:
                    q3["pass"] = False
                    q3["runs"][f"{arm}_s{seed}_rand"] = "缺跑"
                continue
            h = pct(r["eval"]["adv_by_ctype"]["heldout_pair"]["acc"])
            s = pct(r["eval"]["adv_by_ctype"]["shifted_pos"]["acc"])
            e = {"heldout": h, "shifted": s, "ok": h <= limit and s <= limit}
            q3["runs"][f"{arm}_s{seed}_rand"] = e
            # 判据约束臂：臂 A 与过 Q1 的臂
            if arm == "A" or arm in q1["passing"]:
                if not e["ok"]:
                    q3["pass"] = False
    out["Q"]["Q3"] = q3

    # ---- Q5：参数量 --------------------------------------------------------
    q5 = {"per_arm": {}, "max_ratio_vs_A": None}
    for arm in ARMS:
        r = load(arm, 42)
        if r is None:
            continue
        q5["per_arm"][arm] = {"agg_params": r["freeze"]["agg_params"],
                              "head_params": r["freeze"]["head_params"],
                              "trainable": r["freeze"]["trainable_params"],
                              "encoder_trainable": r["freeze"]["encoder_trainable"]}
    if "A" in q5["per_arm"] and q1["passing"]:
        a = q5["per_arm"]["A"]["trainable"]
        ratios = [q5["per_arm"][x]["trainable"] / a for x in q1["passing"]
                  if x in q5["per_arm"]]
        q5["max_ratio_vs_A"] = round(max(ratios), 6) if ratios else None
        q5["over_1p10"] = bool(ratios and max(ratios) > 1.10)
    out["Q"]["Q5"] = q5

    # ---- 判定 --------------------------------------------------------------
    if out["missing"]:
        verdict = "③ 证据不足（有缺跑）"
    elif not q4["pass"]:
        verdict = "③ 证据不足（Q4 臂 A 未复现 B 族：口径不一致）"
    elif not q3["pass"]:
        verdict = "③ 证据不足（Q3 随机标签对照未掉回 50%：判据失效）"
    elif q1["passing"] and q2["pass"]:
        verdict = "① 墙在池化（过 Q1 的臂：" + ", ".join(q1["passing"]) + "）"
    elif q1["passing"]:
        verdict = "③ 证据不足（过 Q1 但 Q2 未全过）"
    else:
        verdict = "② 墙在表示（四臂均未过 Q1，且 Q4、Q3 成立）"
    out["verdict"] = verdict

    # ---- 打印 --------------------------------------------------------------
    print(f"SE(n=1250) = {SE1250*100:.3f}pt，2×SE = {TWO_SE:.3f}pt\n")
    print("| 臂 | seed | heldout_pair | shifted_pos | adv 总 | test | train | 可训参数 |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['arm']} | {row['seed']} | {row['heldout']:.2f} | "
              f"{row['shifted']:.2f} | {row['adv']:.2f} | {row['test']:.2f} | "
              f"{row['train']:.2f} | {row['trainable']} |")
    print("\nQ1（heldout > 臂 A，余量 > 2×SE，两 seed 同号）：")
    for arm, v in q1["per_arm"].items():
        print(f"  {arm}: {json.dumps(v, ensure_ascii=False)}")
    print(f"Q2: {json.dumps(q2, ensure_ascii=False)}")
    print(f"Q3: {json.dumps(q3, ensure_ascii=False)}")
    print(f"Q4: {json.dumps(q4, ensure_ascii=False)}")
    print(f"Q5: {json.dumps(q5, ensure_ascii=False)}")
    print(f"\n判定：{verdict}")

    (RESULTS / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("→ results/summary.json")


if __name__ == "__main__":
    main()
