#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12b 判据计算：E1 / E2 / E3 / E6 + 卡−地板（PREREG §4 + 本单元任务口径）。

输入：results/battery.json（地板，fit=train 的逐条 test 预测）、
      results/train_results.json（卡，2 seed × 3 臂 + 随机标签对照）。
输出：results/ecriteria.json + stdout 表。纯 CPU。

门槛（跑前写死）：
  E1 主：X test acc > max_naive + 2×SE（n=1200 ⇒ SE=1.443pt、2SE=2.886pt），2 seed 同号；
  E2：C / D 逐型 acc > 该型 max_naive + 2×SE（n=300 ⇒ SE=2.887pt、2SE=5.774pt）；
  E3：随机标签 acc ≤ train 多数类 + 2×SE；
  E6：X-cross / X-ir 的 acc（必报；≈ 盲猜 时判「只记住实体」）。
判定三选一：有真能力（E1 过）/ 只记住实体（E6 两臂 ≈ 盲猜）/ 证据不足（其余）。**不许硬选**。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEEDS = (42, 43)
ARMS = ("x", "x-ir", "x-cross")
CHANCE = 0.52886          # majority 0.5 + 2×SE(1200)


def se(n):
    return math.sqrt(0.25 / n)


def main() -> int:
    bat = json.loads((RES / "battery.json").read_text(encoding="utf-8"))
    trn = json.loads((RES / "train_results.json").read_text(encoding="utf-8"))
    runs = trn["runs"]
    out: dict = {"se": {}, "E1": {}, "E2": {}, "E3": {}, "E6": {}, "delta_vs_each": {}}

    # ---- 逐型地板（E2 用）：规则 train 拟合后在该型 test 行上的 acc 取最大 ----
    def subset_floor(arm, key, value):
        b = bat["arms"][arm]
        idx = [i for i, k in enumerate(b[key]) if k == value]
        best, best_rule = 0.0, None
        for r in b["counted_rules"]:
            pr = b["preds"][r]
            a = sum(1 for i in idx if pr[i] == b["y"][i]) / max(len(idx), 1)
            if a > best:
                best, best_rule = a, r
        return best, best_rule, len(idx)

    # ================= E1（主） =================
    for arm in ARMS:
        b = bat["arms"][arm]
        n = b["n_test"]
        gate = b["max_naive"] + 2 * se(n)
        rows = []
        for seed in SEEDS:
            r = next(x for x in runs
                     if x["arm"] == arm and x["seed"] == seed and not x["randlabel"])
            card = r["own_test"]["acc"]
            rows.append({"seed": seed, "card": card, "max_naive": b["max_naive"],
                         "gate": gate, "margin": card - gate,
                         "pass": card > gate})
        out["E1"][arm] = {"n": n, "se": se(n), "max_naive": b["max_naive"],
                          "max_naive_rule": b["max_naive_rule"], "gate": gate,
                          "seeds": rows,
                          "both_same_sign": (rows[0]["pass"] == rows[1]["pass"]),
                          "pass": all(r["pass"] for r in rows)}
        print(f"[E1] {arm:8s} 卡={rows[0]['card']:.4f}/{rows[1]['card']:.4f} "
              f"max_naive={b['max_naive']:.4f}({b['max_naive_rule']}) "
              f"门槛={gate:.4f} ⇒ {'过' if out['E1'][arm]['pass'] else '未过'}"
              f"（卡−地板={rows[0]['card'] - b['max_naive']:+.4f}）")

    # ================= E2（逐型，X 臂） =================
    e2_rows = []
    for typ in ("A", "B", "C", "D"):
        floor, rule, n = subset_floor("x", "type", typ)
        r42 = next(x for x in runs if x["arm"] == "x" and x["seed"] == 42
                   and not x["randlabel"])["own_test"]["per_type"][typ]
        r43 = next(x for x in runs if x["arm"] == "x" and x["seed"] == 43
                   and not x["randlabel"])["own_test"]["per_type"][typ]
        gate = floor + 2 * se(n)
        row = {"type": typ, "n": n, "se": se(n), "floor": floor, "floor_rule": rule,
               "gate": gate, "card_s42": r42["acc"], "card_s43": r43["acc"],
               "pass": r42["acc"] > gate and r43["acc"] > gate}
        e2_rows.append(row)
        print(f"[E2] {typ}: n={n} 卡={r42['acc']:.4f}/{r43['acc']:.4f} "
              f"该型地板={floor:.4f}({rule}) 门槛={gate:.4f} ⇒ "
              f"{'过' if row['pass'] else '未过'}")
    out["E2"] = {"types": e2_rows,
                 "neg_types_pass": all(r["pass"] for r in e2_rows if r["type"] in ("C", "D")),
                 "all_types_pass": all(r["pass"] for r in e2_rows)}

    # ---- X-ir 逐子型（E2 同口径扩展，必报） ----
    ir_rows = []
    for sub in sorted(set(bat["arms"]["x-ir"]["subtype"])):
        floor, rule, n = subset_floor("x-ir", "subtype", sub)
        accs = []
        for seed in SEEDS:
            r = next(x for x in runs if x["arm"] == "x-ir" and x["seed"] == seed
                     and not x["randlabel"])["own_test"]["per_subtype"][sub]
            accs.append(r["acc"])
        ir_rows.append({"subtype": sub, "n": n, "floor": floor, "floor_rule": rule,
                        "gate": floor + 2 * se(n), "card": accs})
        print(f"[E2·x-ir] {sub}: n={n} 卡={accs[0]:.4f}/{accs[1]:.4f} "
              f"地板={floor:.4f}({rule}) 门槛={floor + 2 * se(n):.4f}")
    out["E2"]["x_ir_subtypes"] = ir_rows

    # ================= E3（随机标签） =================
    e3 = []
    for seed in SEEDS:
        r = next(x for x in runs if x["arm"] == "x" and x["seed"] == seed and x["randlabel"])
        acc = r["own_test"]["acc"]
        maj = bat["arms"]["x"]["rules"]["majority"]["acc"]
        gate = maj + 2 * se(r["own_test"]["n"])
        e3.append({"seed": seed, "acc": acc, "majority": maj, "gate": gate,
                   "pass": acc <= gate})
        print(f"[E3] seed={seed} 随机标签 acc={acc:.4f} ≤ {gate:.4f} ⇒ "
              f"{'过' if acc <= gate else '未过'}")
    out["E3"] = {"runs": e3, "pass": all(r["pass"] for r in e3)}

    # ================= E6（X-cross / X-ir） =================
    for arm in ("x-cross", "x-ir"):
        b = bat["arms"][arm]
        own, cross = [], []
        for seed in SEEDS:
            own.append(next(x for x in runs if x["arm"] == arm and x["seed"] == seed
                            and not x["randlabel"])["own_test"]["acc"])
            cross.append(next(x for x in runs if x["arm"] == "x" and x["seed"] == seed
                              and not x["randlabel"])["cross_eval"][arm]["acc"])
        out["E6"][arm] = {"own_trained": own, "x_trained": cross,
                          "max_naive": b["max_naive"],
                          "blind_guess_gate": CHANCE,
                          "approx_blind": all(a <= CHANCE for a in own)}
        print(f"[E6] {arm}: 自训 acc={own[0]:.4f}/{own[1]:.4f} "
              f"X 臂训 acc={cross[0]:.4f}/{cross[1]:.4f} "
              f"地板={b['max_naive']:.4f} 盲猜线={CHANCE}")

    # ================= 卡 − 每条地板（三臂 × 9 条） =================
    for arm in ARMS:
        b = bat["arms"][arm]
        card = [next(x for x in runs if x["arm"] == arm and x["seed"] == s
                     and not x["randlabel"])["own_test"]["acc"] for s in SEEDS]
        out["delta_vs_each"][arm] = {
            r: [round(c - b["rules"][r]["acc_test"], 6) for c in card]
            for r in b["counted_rules"]}

    # ================= 判定三选一 =================
    e1_pass = out["E1"]["x"]["pass"] and out["E1"]["x"]["both_same_sign"]
    blind = all(out["E6"][a]["approx_blind"] for a in ("x-cross", "x-ir"))
    if e1_pass:
        verdict = "有真能力（E1 过）"
    elif blind:
        verdict = "只记住实体（E6 两臂 ≈ 盲猜）"
    else:
        verdict = "证据不足"
    out["verdict"] = verdict
    out["E1_main_pass"] = e1_pass
    print(f"\n判定（三选一）：{verdict}")
    print(f"  E1 主臂 {'过' if e1_pass else '未过'}；"
          f"E2 {'过' if out['E2']['neg_types_pass'] else '未过'}；"
          f"E3 {'过' if out['E3']['pass'] else '未过'}；"
          f"E6 两臂≈盲猜={blind}")
    (RES / "ecriteria.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"→ {RES / 'ecriteria.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
