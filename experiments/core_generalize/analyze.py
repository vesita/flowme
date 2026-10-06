"""core_generalize 汇总：δ、主表、P1/P2/P3、健康检查、判定。

    uv run python experiments/core_generalize/analyze.py
只读本目录产物 + core_keep/capability_map 的参考数字；写 summary.json / summary.md。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBES = HERE / "probes"
CORES = HERE / "cores"
ARMS = ("C1", "C3", "C5", "C1x5")
SEEDS = (42, 43)


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    info = load_json(HERE / "data_info.json")
    holdouts = [l.strip() for l in (HERE / "holdouts.txt").read_text().splitlines()
                if l.strip()]

    # ---- 探针结果 ----
    P: dict[tuple[str, str, int], dict] = {}
    for f in sorted(PROBES.glob("*.json")):
        r = load_json(f)
        P[(r["label"], r["holdout"], r["seed"])] = r

    floors = {h: {s: info["tasks"][h]["splits"][str(s)]["floor_exact_empty_predictor"]
                  for s in SEEDS} for h in holdouts}

    # ---- δ：max(2pt, B0 两头 seed 极差) —— 同一公式，跑前已写死 ----
    delta = {}
    for h in holdouts:
        vals = [P[("B0", h, s)]["metrics"]["exact_match"] for s in SEEDS
                if ("B0", h, s) in P]
        rng = max(vals) - min(vals) if len(vals) == 2 else float("nan")
        delta[h] = max(0.02, rng)
        print(f"[delta] {h}: B0 exact s42/s43 = "
              f"{' / '.join(f'{v:.4f}' for v in vals)} 极差={rng:.4f} "
              f"⇒ δ={delta[h]:.4f} ({delta[h]*100:.2f}pt)")
    delta_global = max(delta.values())
    print(f"[delta] δ_global = {delta_global:.4f} ({delta_global*100:.2f}pt)")

    def ex(label: str, h: str, s: int) -> float | None:
        r = P.get((label, h, s))
        return None if r is None else r["metrics"]["exact_match"]

    # ---- 主表 ----
    table_rows = []
    for h in holdouts:
        for s in SEEDS:
            row = {"holdout": h, "seed": s, "floor": floors[h][s],
                   "delta": delta[h],
                   "B0": ex("B0", h, s), "C1": ex(f"C1_s{s}", h, s),
                   "C3": ex(f"C3_s{s}", h, s), "C5": ex(f"C5_s{s}", h, s),
                   "C1x5": ex("C1x5_s42", h, s) if s == 42 else None}
            table_rows.append(row)

    # ---- 贴地板标记 ----
    for row in table_rows:
        lim = row["floor"] + row["delta"]
        row["floor_flag"] = {k: (row[k] is not None and row[k] <= lim)
                             for k in ("B0", "C1", "C3", "C5", "C1x5")}

    # ---- 判据 ----
    verdicts = {}
    for h in holdouts:
        d = delta[h]
        rows = {r["seed"]: r for r in table_rows if r["holdout"] == h}
        p1 = all(rows[s]["C5"] is not None and rows[s]["C1"] is not None
                 and rows[s]["C5"] >= rows[s]["C1"] + d for s in SEEDS)
        p2 = all(rows[s]["C3"] is not None and rows[s]["C5"] is not None
                 and rows[s]["C3"] >= rows[s]["C1"] - d
                 and rows[s]["C5"] >= rows[s]["C3"] - d
                 and rows[s]["C5"] >= rows[s]["C1"] for s in SEEDS)
        r42 = rows[42]
        p3 = (r42["C5"] is not None and r42["C1x5"] is not None
              and r42["C5"] >= r42["C1x5"] + d)
        data_explained = (r42["C5"] is not None and r42["C1x5"] is not None
                          and abs(r42["C5"] - r42["C1x5"]) < d)
        d1 = {s: (rows[s]["C5"] - rows[s]["C1"]) for s in SEEDS}
        # 「两 seed 同号」的两种读法都算（PREREG 写的是同号，没规定 0.0000 算什么）：
        #   strict = 严格同号（>0 都为真 / 都为假）；noconflict = 只要两 seed 不反号即可。
        consistent = (d1[42] > 0) == (d1[43] > 0)
        no_conflict = not (d1[42] * d1[43] < 0)
        core_cells = [rows[s][k] for s in SEEDS for k in ("C1", "C3", "C5")
                      if rows[s][k] is not None]
        arms_floor = all(rows[s][k] is not None and rows[s][k] <= rows[s]["floor"] + d
                         for s in SEEDS for k in ("C1", "C3", "C5"))
        b0_floor = any(rows[s]["B0"] is not None and rows[s]["B0"] <= rows[s]["floor"] + d
                       for s in SEEDS)
        verdicts[h] = {"P1": p1, "P2": p2, "P3": p3, "data_explained_C1x5": data_explained,
                       "C5_minus_C1": d1, "consistent_2seed": consistent,
                       "no_sign_conflict": no_conflict,
                       "all_core_arms_floor": arms_floor, "B0_at_floor": b0_floor,
                       "n_core_cells": len(core_cells)}

    # ---- 健康检查（集内卡 vs core_keep J0 同 seed 同任务）----
    health = {}
    j0 = {}
    for s in SEEDS:
        p = ROOT / "experiments" / "core_keep" / "cards" / f"J0_s{s}_metrics.json"
        if p.exists():
            j0[s] = load_json(p)["metrics"]
    voided = set()
    for arm in ARMS:
        health[arm] = {}
        for s in SEEDS:
            f = CORES / f"{arm}_s{s}_metrics.json"
            if not f.exists():
                continue
            m = load_json(f)
            rec = {}
            for task, mv in m["metrics"].items():
                ref = j0.get(s, {}).get(task, {}).get("exact_match")
                rec[task] = {"exact": mv["exact_match"], "j0": ref,
                             "delta_vs_j0": None if ref is None
                             else mv["exact_match"] - ref}
            health[arm][s] = rec
        # 作废线：任一集内任务比 J0 低 ≥ δ_global，且两 seed 同向
        for task in {t for sd in health[arm].values() for t in sd}:
            ds = [health[arm][s].get(task, {}).get("delta_vs_j0")
                  for s in SEEDS if s in health[arm] and task in health[arm].get(s, {})]
            ds = [d for d in ds if d is not None]
            if ds and all(d is not None and d <= -delta_global for d in ds):
                voided.add(arm)

    # ---- 判定（预注册三选一，条件见 PREREG §5）----
    def decide(key: str):
        """key = 'consistent_2seed'（严格同号）或 'no_sign_conflict'（不反号即可）。
        PREREG 只写了「两 seed 同号才下结论」，0.0000 算什么没规定 ⇒ 两种读法都报。"""
        usable = [h for h in holdouts
                  if verdicts[h][key] and not verdicts[h]["all_core_arms_floor"]]
        dropped = {h: ("两 seed 不同号" if not verdicts[h][key] else "三个核臂全贴地板")
                   for h in holdouts if h not in usable}
        if len(usable) < 2:
            return ("证据不足",
                    f"可用 hold-out 只有 {len(usable)} 个（<2）：{usable}；"
                    f"被剔除：{json.dumps(dropped, ensure_ascii=False)}", usable, dropped)
        if voided:
            return ("证据不足", f"健康检查作废了核臂 {sorted(voided)}", usable, dropped)
        p1 = [verdicts[h]["P1"] for h in usable]
        p2 = [verdicts[h]["P2"] for h in usable]
        p3 = [verdicts[h]["P3"] for h in usable]
        if all(p1) and all(p3) and all(p2):
            return ("a 核在变通用",
                    "可用 hold-out 上 P1/P2/P3 全过（两 seed 同号、两 hold-out 同向）",
                    usable, dropped)
        if not any(p1):
            return ("b 表示本来够用",
                    "可用 hold-out 上 P1 全不过（C5−C1 < δ）⇒ 提升不是多样性带来的",
                    usable, dropped)
        if all(p1) and not any(p3):
            return ("b 表示本来够用（数据/优化解释）",
                    "P1 全过但 P3 全不过"
                    + ("（C1x5 ≈ C5）" if all(verdicts[h]["data_explained_C1x5"]
                                              for h in usable) else "")
                    + " ⇒ 提升可由「更多数据/优化」解释，不是多样性",
                    usable, dropped)
        return ("证据不足",
                "两 seed 不同号 / 两 hold-out 方向不一致 / 判据组合不落在预注册三类里："
                + json.dumps({h: {k: verdicts[h][k] for k in
                                  ("P1", "P2", "P3", "consistent_2seed",
                                   "no_sign_conflict", "all_core_arms_floor")}
                              for h in holdouts}, ensure_ascii=False),
                usable, dropped)

    verdict, why, usable, dropped = decide("consistent_2seed")
    verdict_nc, why_nc, usable_nc, dropped_nc = decide("no_sign_conflict")

    # ---- 锚点（描述性，非判据）：各核相对未训练 base 核（B0）的差 ----
    anchor = {}
    for h in holdouts:
        for s in SEEDS:
            b = ex("B0", h, s)
            anchor[f"{h}_s{s}"] = {
                "B0": b,
                **{k: (None if (v is None or b is None) else v - b)
                   for k, v in (("C1", ex(f"C1_s{s}", h, s)),
                                ("C3", ex(f"C3_s{s}", h, s)),
                                ("C5", ex(f"C5_s{s}", h, s)),
                                ("C1x5", ex("C1x5_s42", h, s) if s == 42 else None))}}

    out = {"delta": delta, "delta_global": delta_global, "holdouts": holdouts,
           "floors": floors, "table": table_rows, "criteria": verdicts,
           "health": health, "voided_arms": sorted(voided),
           "anchor_vs_B0": anchor,
           "usable_holdouts": usable, "dropped_holdouts": dropped,
           "verdict": verdict, "verdict_why": why,
           "verdict_alt_noconflict": verdict_nc, "verdict_alt_why": why_nc,
           "usable_holdouts_alt": usable_nc}
    (HERE / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict(严格同号)": verdict, "why": why,
                      "verdict(不反号即可)": verdict_nc, "why_alt": why_nc,
                      "delta": delta, "usable": usable, "dropped": dropped,
                      "voided": sorted(voided)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
