#!/usr/bin/env python3
"""汇总 C0–C5 ⇒ `results/summary.json`，并打印报告用的表。

    uv run python experiments/counterfactual/analyze.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from common import RESULTS
from perturb import FAMILIES, FAMILY_NAME


def q(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)


def pair_delta(pairs: list[tuple[int, int]]) -> dict:
    """配对二值：结构化是否找到 vs 随机是否找到。Δ=(b−c)/n，SE=sqrt((b+c)−(b−c)^2/n)/n。"""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "p_struct": None, "p_rand": None, "delta": None,
                "se": None, "b": 0, "c": 0, "sig": None}
    b = sum(1 for s, r in pairs if s == 1 and r == 0)     # 仅结构化找到
    c = sum(1 for s, r in pairs if s == 0 and r == 1)     # 仅随机找到
    d = (b - c) / n
    se = ((b + c) - (b - c) ** 2 / n) ** 0.5 / n if n else 0.0
    ps = sum(s for s, _ in pairs) / n
    pr = sum(r for _, r in pairs) / n
    return {"n": n, "p_struct": round(ps, 4), "p_rand": round(pr, 4),
            "delta": round(d, 4), "se": round(se, 4),
            "b": b, "c": c, "sig": bool(d > 2 * se) if n else None,
            "t": round(d / se, 2) if se > 0 else None}


def cat(rec: dict) -> str:
    cb, ca = rec["ch_before"], rec["ch_after"]
    if cb == "ok" and ca == "ok":
        return "ok→ok 文本变"
    if cb == "reject" and ca == "ok":
        return "reject→ok"
    if cb == "ok" and ca == "reject":
        return "ok→reject"
    return "reject→reject 拒因变"


def main() -> int:
    S = json.loads((RESULTS / "search.json").read_text(encoding="utf-8"))
    V = json.loads((RESULTS / "verify.json").read_text(encoding="utf-8"))
    base = {b["idx"]: b for b in S["baseline"]}

    out: dict = {}
    out["baseline"] = {
        "n": len(base),
        "ok": sum(1 for b in base.values() if b["ch"] == "ok"),
        "reject": sum(1 for b in base.values() if b["ch"] == "reject"),
        "reasons": dict(Counter(b["y"][2] for b in base.values() if b["ch"] == "reject")),
    }

    # ---- C0 / C3 ----
    out["C0"] = {"struct_fail": V["struct_fail"], "struct_checked": V["struct_checked"],
                 "rate": round(V["struct_fail"] / max(1, V["struct_checked"]), 4),
                 "R1_fail": V["R1_fail"], "R1_checked": V["R1_checked"],
                 "R2_fail": V["R2_fail"], "R2_checked": V["R2_checked"],
                 "baseline_mismatch": V["baseline_mismatch"],
                 "n_fail_total": len(V["claims"])}
    out["C3"] = {"baseline_mismatch": V["baseline_mismatch"],
                 "claim_mismatch": len(V["claims"]),
                 "consistent": V["baseline_mismatch"] == 0 and len(V["claims"]) == 0}

    # ---- 逐族聚合 ----
    per = {}
    pairs_all: dict[str, list] = {m: [] for m in ("R1", "R2", "RUNION")}
    sizes_all: dict[str, list[int]] = defaultdict(list)
    per_cell = []
    n_unreachable = {"R1": 0, "R2": 0}
    n_rand_draws = {"R1": 0, "R2": 0}
    n_rand_hit = {"R1": 0, "R2": 0}
    n_struct_draws = n_struct_hit = 0
    cats: dict[str, Counter] = {k: Counter() for k in ("struct", "R1", "R2")}
    examples: list[dict] = []

    for rec in S["search"]:
        fam = rec["family"]
        if rec.get("budget_K", 0) == 0:
            continue
        st = rec["struct"]
        s_found = 1 if st.get("found") else 0
        chb = base[rec["idx"]]["ch"]
        def _ch(runs):            # 辅口径：任一次跑的通道(ok/reject)发生切换
            return 1 if any(len(x) > 3 and x[3][0] != chb for x in runs) else 0
        s_ch = _ch(st["runs"])
        r1_ch = _ch(rec["R1"]["runs"])
        r2_ch = _ch(rec["R2"]["runs"])
        r = {}
        for m in ("R1", "R2"):
            f = rec[m]
            r[m] = 1 if f.get("found") else 0
            n_unreachable[m] += f.get("unreachable", 0)
        # 实际跑批数：结构化/随机各自的前向次数（不可达的抽样不计前向）
        per_cell.append((fam, s_found, r["R1"], r["R2"],
                         st["n_run"], rec["R1"]["n_run"], rec["R2"]["n_run"],
                         sum(1 for x in rec["R1"]["runs"] if x[2] == 1),
                         sum(1 for x in rec["R2"]["runs"] if x[2] == 1),
                         s_ch, r1_ch, r2_ch))

        for m in ("R1", "R2", "RUNION"):
            pairs_all[m].append((s_found, r["R1"] if m == "R1" else
                                 (r["R2"] if m == "R2" else max(r["R1"], r["R2"]))))
        if s_found:
            sizes_all[fam].append(st["found"]["lev"])
            sizes_all["ALL"].append(st["found"]["lev"])
            cats["struct"][cat(st["found"])] += 1
            if len(examples) < 400:
                examples.append({"kind": "struct", **{k: st["found"][k] for k in
                                                      ("idx", "family", "lev", "x",
                                                       "x_prime", "y", "y_prime",
                                                       "ch_before", "ch_after")}})
        for m in ("R1", "R2"):
            if rec[m].get("found"):
                cats[m][cat(rec[m]["found"])] += 1

    for fam in FAMILIES:
        cells = [c for c in per_cell if c[0] == fam]
        if not cells:
            per[fam] = {"name": FAMILY_NAME[fam], "cells": 0}
            continue
        struct_draws = sum(c[4] for c in cells)
        struct_hits = sum(1 for c in cells if c[1])
        st_runs_hit = 0
        for rec in S["search"]:
            if rec["family"] == fam and rec.get("budget_K", 0):
                st_runs_hit += sum(1 for x in rec["struct"]["runs"] if x[2] == 1)
        pr = {m: pair_delta([(c[1], c[2 if m == "R1" else 3]) for c in cells])
              for m in ("R1", "R2")}
        per[fam] = {
            "name": FAMILY_NAME[fam], "cells": len(cells),
            "struct_found": struct_hits,
            "struct_per_draw": {"hit": st_runs_hit, "n": struct_draws,
                                "p": round(st_runs_hit / max(1, struct_draws), 4)},
            "rand_per_draw": {
                "R1": {"hit": sum(c[7] for c in cells), "n": sum(c[5] for c in cells),
                       "p": round(sum(c[7] for c in cells) / max(1, sum(c[5] for c in cells)), 4)},
                "R2": {"hit": sum(c[8] for c in cells), "n": sum(c[6] for c in cells),
                       "p": round(sum(c[8] for c in cells) / max(1, sum(c[6] for c in cells)), 4)}},
            "paired": pr,
            "size": {"n": len(sizes_all[fam]), "p25": q(sizes_all[fam], .25),
                     "p50": q(sizes_all[fam], .5), "p75": q(sizes_all[fam], .75),
                     "max": max(sizes_all[fam]) if sizes_all[fam] else 0,
                     "min": min(sizes_all[fam]) if sizes_all[fam] else 0},
        }

    # ---- C1 合计（F1–F4 入判定；F5 单列，样本量过小） ----
    def total(fams: tuple, mode: str) -> dict:
        cells = [c for c in per_cell if c[0] in fams]
        col = 2 if mode == "R1" else 3
        return pair_delta([(c[1], c[col]) for c in cells])

    out["C1"] = {
        "F1_F4": {m: total(FAMILIES[:4], m) for m in ("R1", "R2")},
        "F5": {m: total(("F5",), m) for m in ("R1", "R2")},
        "ALL5": {m: total(FAMILIES, m) for m in ("R1", "R2")},
        "RUNION_F1_F4": pair_delta([(c[1], max(c[2], c[3])) for c in per_cell
                                    if c[0] in FAMILIES[:4]]),
        "channel_only_F1_F4": {          # 辅口径：只看 ok↔reject 通道切换（不入判定）
            "R1": pair_delta([(c[9], c[10]) for c in per_cell if c[0] in FAMILIES[:4]]),
            "R2": pair_delta([(c[9], c[11]) for c in per_cell if c[0] in FAMILIES[:4]]),
            "struct_cells_hit": sum(c[9] for c in per_cell if c[0] in FAMILIES[:4]),
        },
        "per_draw": {
            "struct": {"hit": sum(1 for rec in S["search"] for x in
                                  rec.get("struct", {}).get("runs", []) if len(x) > 3 and x[2]),
                       "n": sum(rec.get("struct", {}).get("n_run", 0) for rec in S["search"])},
        },
    }
    st_hit = sum(1 for rec in S["search"] for x in rec.get("struct", {}).get("runs", [])
                 if x[2] == 1)
    st_n = sum(rec.get("struct", {}).get("n_run", 0) for rec in S["search"])
    out["C1"]["per_draw"]["struct"] = {"hit": st_hit, "n": st_n,
                                       "p": round(st_hit / max(1, st_n), 4)}
    for m in ("R1", "R2"):
        hit = sum(1 for rec in S["search"] for x in rec.get(m, {}).get("runs", [])
                  if x[2] == 1)
        n = sum(rec.get(m, {}).get("n_run", 0) for rec in S["search"])
        out["C1"]["per_draw"][m] = {"hit": hit, "n": n, "p": round(hit / max(1, n), 4)}
    out["unreachable"] = {m: sum(rec.get(m, {}).get("unreachable", 0)
                                 for rec in S["search"]) for m in ("R1", "R2")}
    out["C1"]["n_cells"] = len(per_cell)
    out["C1"]["n_cells_F5"] = sum(1 for c in per_cell if c[0] == "F5")

    # ---- C2 ----
    out["C2"] = {fam: per[fam].get("size") for fam in FAMILIES}
    out["C2"]["ALL"] = {"n": len(sizes_all["ALL"]), "p25": q(sizes_all["ALL"], .25),
                        "p50": q(sizes_all["ALL"], .5), "p75": q(sizes_all["ALL"], .75),
                        "max": max(sizes_all["ALL"]) if sizes_all["ALL"] else 0}
    # 每样本全局最小
    gmin = []
    for i, b in base.items():
        ds = [rec["struct"]["found"]["lev"] for rec in S["search"]
              if rec["idx"] == i and rec.get("struct", {}).get("found")]
        if ds:
            gmin.append(min(ds))
    out["C2"]["global_min"] = {"n_samples_with_cf": len(gmin),
                               "p25": q(gmin, .25), "p50": q(gmin, .5),
                               "p75": q(gmin, .75), "max": max(gmin) if gmin else 0}

    # ---- C4 ----
    out["C4"] = {k: dict(v) for k, v in cats.items()}
    out["C4"]["struct_channel_change"] = sum(
        cats["struct"][k] for k in ("reject→ok", "ok→reject"))
    out["C4"]["struct_ok_text_change"] = cats["struct"]["ok→ok 文本变"]
    out["per_family"] = per
    out["examples"] = examples

    # ---- C5 判定 ----
    c0_ok = out["C0"]["rate"] <= 0.05 and out["C0"]["struct_checked"] >= 50
    c3_ok = out["C3"]["consistent"]
    unreach_rate = max(out["unreachable"].values()) / max(1, sum(
        rec.get(m, {}).get("n_run", 0) for rec in S["search"] for m in ("R1", "R2")))
    d1, d2 = out["C1"]["F1_F4"]["R1"], out["C1"]["F1_F4"]["R2"]
    sig = bool(d1["sig"]) and bool(d2["sig"])
    enough = out["C0"]["struct_checked"] >= 50
    if c0_ok and c3_ok and enough and sig:
        verdict = "反事实真实且特异"
    elif c0_ok and c3_ok and enough:
        verdict = "真实但不特异"
    else:
        verdict = "证据不足"
    out["verdict"] = verdict
    out["verdict_inputs"] = {"c0_rate": out["C0"]["rate"], "c0_n": out["C0"]["struct_checked"],
                             "c3_consistent": c3_ok, "delta_R1": d1, "delta_R2": d2,
                             "rand_unreachable": out["unreachable"]}

    path = RESULTS / "summary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("baseline", "C0", "C3", "C1", "C2", "C4", "verdict")},
                     ensure_ascii=False, indent=1)[:6000])
    print(f"\n[写出] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
