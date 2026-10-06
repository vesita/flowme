#!/usr/bin/env python3
"""layer_selective 判据计算（PREREG §3 X0–X5 + 三选一判定）→ results/summary.json。

用法：uv run python experiments/layer_selective/analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
SEL = ROOT / "experiments" / "select_pool" / "results"          # 只读（F 的 X0 参照）
CSO = ROOT / "experiments" / "core_select_only" / "results"     # 只读（ALL 的 X0 参照）

ARMS = ("F", "ALL", "EMB", "TOP", "ADPT")
SEEDS = (42, 43)
SE = math.sqrt(0.25 / 1250)
TWO_SE = 2 * SE                                    # 0.0282842...
BAND = {"pronoun": 0.0283, "sentiment": 0.0041, "relation": 0.0139, "person": 0.0033}
OLD = list(BAND)

# ---- X0 目标（PREREG §3 写死，来自 select_pool 臂 A 与 core_select_only::selonly）----
X0_F = {
    42: {"heldout_pair": 49.52, "shifted_pos": 92.56, "test": 94.48, "train": 100.00,
         "adv": 71.04, "loss_first": 0.693002, "core_drift": 0.0},
    43: {"heldout_pair": 51.04, "shifted_pos": 92.48, "test": 94.48, "train": 100.00,
         "adv": 71.76, "loss_first": 0.692196, "core_drift": 0.0},
}
X0_ALL = {
    42: {"heldout_pair": 73.60, "shifted_pos": 100.00, "test": 100.00, "train": 100.00,
         "adv": 86.80, "loss_first": 0.693002, "loss_select_first50": 0.153912,
         "core_drift": 0.05003196568723828,
         "r5": {"pronoun": -0.461667, "sentiment": -0.485625,
                "relation": -0.679167, "person": -0.005}},
    43: {"heldout_pair": 68.40, "shifted_pos": 100.00, "test": 99.96, "train": 100.00,
         "adv": 84.20, "loss_first": 0.692196, "loss_select_first50": 0.134212,
         "core_drift": 0.048874110162901115,
         "r5": {"pronoun": -0.515, "sentiment": -0.502188,
                "relation": -0.694444, "person": -0.006667}},
}


def load(arm: str, seed: int, randlabel: bool = False) -> dict | None:
    p = RESULTS / f"{arm}_s{seed}{'_rand' if randlabel else ''}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def ctype(d: dict, k: str) -> float:
    return pct(d["eval"]["adv_by_ctype"][k]["acc"])


def ev(d: dict, k: str) -> float:
    return pct(d["eval"][k]["acc"])


def main() -> None:
    out: dict = {"se": round(SE, 6), "two_se_pt": round(100 * TWO_SE, 4), "arms": {}}
    missing = []

    # ================= X0（门禁）=================
    x0_rows, x0_ok = [], True
    for seed in SEEDS:
        f, a = load("F", seed), load("ALL", seed)
        if f is None or a is None:
            missing.append(f"F/ALL s{seed}")
            x0_ok = False
            continue
        tgt_f = dict(X0_F[seed])
        got_f = {"heldout_pair": ctype(f, "heldout_pair"), "shifted_pos": ctype(f, "shifted_pos"),
                 "test": ev(f, "test"), "train": ev(f, "train"), "adv": ev(f, "adv"),
                 "loss_first": f["loss_first"], "core_drift": f["core_drift"]}
        for k, v in tgt_f.items():
            d = round(got_f[k] - v, 6)
            x0_rows.append({"arm": "F", "seed": seed, "item": k, "target": v,
                            "got": got_f[k], "delta": d})
            if d != 0.0:
                x0_ok = False
        tgt_a = dict(X0_ALL[seed])
        r5t = tgt_a.pop("r5")
        got_a = {"heldout_pair": ctype(a, "heldout_pair"), "shifted_pos": ctype(a, "shifted_pos"),
                 "test": ev(a, "test"), "train": ev(a, "train"), "adv": ev(a, "adv"),
                 "loss_first": a["loss_first"], "loss_select_first50": a["loss_select_first50"],
                 "core_drift": a["core_drift"]}
        for k, v in tgt_a.items():
            d = round(got_a[k] - v, 6)
            x0_rows.append({"arm": "ALL", "seed": seed, "item": k, "target": v,
                            "got": got_a[k], "delta": d})
            if d != 0.0:
                x0_ok = False
        for c, v in r5t.items():
            d = round(a["r5"][c]["delta"] - v, 6)
            x0_rows.append({"arm": "ALL", "seed": seed, "item": f"r5.{c}", "target": v,
                            "got": a["r5"][c]["delta"], "delta": d})
            if d != 0.0:
                x0_ok = False
    out["X0"] = {"pass": x0_ok, "n_items": len(x0_rows),
                 "max_abs_delta": max((abs(r["delta"]) for r in x0_rows), default=None),
                 "rows": x0_rows}

    # ================= 逐臂主表 =================
    arm_stats = {}
    for arm in ARMS:
        rec: dict = {"real": {}, "rand": {}}
        for seed in SEEDS:
            d = load(arm, seed)
            if d is None:
                missing.append(f"{arm} s{seed}")
                continue
            rec["real"][seed] = {
                "heldout": ctype(d, "heldout_pair"), "shifted": ctype(d, "shifted_pos"),
                "test": ev(d, "test"), "train": ev(d, "train"), "adv": ev(d, "adv"),
                "core_drift": d["core_drift"],
                "n_trainable": d["inventory"]["n_trainable"],
                "encoder_trainable": d["inventory"]["encoder_trainable"],
                "adapter_params": d["inventory"]["adapter_params"],
                "sec_per_step_steady": d["timing"]["sec_per_step_steady"],
                "peak_mem_mb": d["timing"]["peak_mem_mb"],
                "wall_sec": d["wall_sec"],
                "r5": {c: d["r5"][c]["delta"] for c in OLD},
                "loss_first": d["loss_first"]}
        for seed in SEEDS:
            d = load(arm, seed, randlabel=True)
            if d is None:
                missing.append(f"{arm} s{seed} rand")
                continue
            rec["rand"][seed] = {"heldout": ctype(d, "heldout_pair"),
                                 "shifted": ctype(d, "shifted_pos"),
                                 "test": ev(d, "test"), "train": ev(d, "train"),
                                 "randlabel_n_diff": d["randlabel_n_diff"]}
        arm_stats[arm] = rec
    out["arms"] = arm_stats

    # X0 参照是否与只读参照文件一致（额外对账，不参与判定）
    ref_chk = []
    for seed in SEEDS:
        p = SEL / f"A_s{seed}.json"
        a = CSO / f"selonly_s{seed}.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            ref_chk.append({"ref": f"select_pool/A_s{seed}", "heldout": pct(r["eval"]["adv_by_ctype"]["heldout_pair"]["acc"])})
        if a.exists():
            r = json.loads(a.read_text(encoding="utf-8"))
            ref_chk.append({"ref": f"core_select_only/selonly_s{seed}", "heldout": pct(r["eval"]["adv_by_ctype"]["heldout_pair"]["acc"])})
    out["reference_files"] = ref_chk

    # ================= X1 新能力 =================
    x1, x1_detail = True, {}
    for arm in ARMS:
        if arm == "ALL":
            x1_detail[arm] = {"note": "X1 的门槛臂本身"}
            continue
        per, ok = {}, True
        for seed in SEEDS:
            if seed not in arm_stats[arm]["real"] or seed not in arm_stats["ALL"]["real"]:
                ok = False
                per[seed] = {"missing": True}
                continue
            thr = arm_stats["ALL"]["real"][seed]["heldout"] - 100 * TWO_SE
            got = arm_stats[arm]["real"][seed]["heldout"]
            per[seed] = {"got": got, "threshold": round(thr, 4),
                         "pass": got >= thr, "margin": round(got - thr, 4)}
            if not (got >= thr):
                ok = False
        x1_detail[arm] = per
        if not ok:
            x1 = False
    out["X1"] = {"pass": x1, "detail": x1_detail}

    # ================= X2 老卡 =================
    x2, x2_detail = True, {}
    for arm in ARMS:
        per, ok = {}, True
        for seed in SEEDS:
            if seed not in arm_stats[arm]["real"]:
                ok = False
                per[seed] = {"missing": True}
                continue
            cells = {}
            for c in OLD:
                d = arm_stats[arm]["real"][seed]["r5"][c]
                cells[c] = {"delta": round(d, 6), "band": BAND[c],
                            "pass": d >= -BAND[c]}
                if not (d >= -BAND[c]):
                    ok = False
            per[seed] = cells
        x2_detail[arm] = per
        if not ok:
            x2 = False
    out["X2"] = {"pass": x2, "detail": x2_detail}

    # ================= X3 shifted =================
    x3, x3_detail = True, {}
    for arm in ARMS:
        if arm == "F":
            x3_detail[arm] = {"note": "门槛臂本身"}
            continue
        per, ok = {}, True
        for seed in SEEDS:
            if seed not in arm_stats[arm]["real"] or seed not in arm_stats["F"]["real"]:
                ok = False
                per[seed] = {"missing": True}
                continue
            thr = arm_stats["F"]["real"][seed]["shifted"] - 100 * TWO_SE
            got = arm_stats[arm]["real"][seed]["shifted"]
            per[seed] = {"got": got, "threshold": round(thr, 4), "pass": got >= thr}
            if not (got >= thr):
                ok = False
        x3_detail[arm] = per
        if not ok:
            x3 = False
    out["X3"] = {"pass": x3, "detail": x3_detail}

    # ================= X4 随机标签（门禁）=================
    x4, x4_detail = True, {}
    for arm in ARMS:
        per, ok = {}, True
        for seed in SEEDS:
            if seed not in arm_stats[arm]["rand"]:
                ok = False
                per[seed] = {"missing": True}
                continue
            r = arm_stats[arm]["rand"][seed]
            h_ok, s_ok = r["heldout"] <= 52.83, r["shifted"] <= 52.83
            per[seed] = {"heldout": r["heldout"], "shifted": r["shifted"],
                         "test": r["test"], "train": r["train"],
                         "n_diff": r["randlabel_n_diff"],
                         "pass": bool(h_ok and s_ok)}
            if not (h_ok and s_ok):
                ok = False
        x4_detail[arm] = per
        if not ok:
            x4 = False
    out["X4"] = {"pass": x4, "detail": x4_detail}

    # ================= X5 必报 =================
    out["X5"] = {arm: {**({} if not arm_stats[arm]["real"] else {
        "n_trainable": arm_stats[arm]["real"][SEEDS[0]]["n_trainable"],
        "encoder_trainable": arm_stats[arm]["real"][SEEDS[0]]["encoder_trainable"],
        "adapter_params": arm_stats[arm]["real"][SEEDS[0]]["adapter_params"],
        "core_drift": {s: arm_stats[arm]["real"][s]["core_drift"] for s in arm_stats[arm]["real"]},
        "sec_per_step_steady": {s: arm_stats[arm]["real"][s]["sec_per_step_steady"]
                                for s in arm_stats[arm]["real"]},
        "peak_mem_mb": {s: arm_stats[arm]["real"][s]["peak_mem_mb"]
                        for s in arm_stats[arm]["real"]},
        "wall_sec": {s: arm_stats[arm]["real"][s]["wall_sec"] for s in arm_stats[arm]["real"]}})}
                 for arm in ARMS}

    # ================= 判定三选一（PREREG §3）=================
    gates_ok = x0_ok and x4 and not missing
    winners = [a for a in ARMS
               if a != "ALL" and a in x1_detail and a in x2_detail
               and x1_detail.get(a) and all(
                   (isinstance(x1_detail[a].get(s), dict) and x1_detail[a][s].get("pass"))
                   for s in SEEDS)
               and all((isinstance(x2_detail[a].get(s), dict)
                        and all(x2_detail[a][s][c]["pass"] for c in OLD))
                       for s in SEEDS if isinstance(x2_detail[a].get(s), dict))
               and all(s in x1_detail[a] and s in x2_detail[a] for s in SEEDS)]
    if missing or not gates_ok:
        verdict, why = "③ 证据不足", "门禁 X0/X4 不过或有缺跑"
    elif winners:
        verdict, why = "① 解耦成功", f"存在同时满足 X1∧X2 的臂：{winners}"
    else:
        verdict, why = "② 解耦不成立", "门禁过，但没有任何臂同时满足 X1∧X2"
    out["verdict"] = {"verdict": verdict, "why": why, "winners": winners,
                      "gates": {"X0": x0_ok, "X4": x4}, "missing": missing}
    out["missing"] = missing

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "missing": missing,
                      "X0": {"pass": x0_ok, "max_abs_delta": out["X0"]["max_abs_delta"]},
                      "X1": x1, "X2": x2, "X3": x3, "X4": x4}, ensure_ascii=False, indent=2))
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
