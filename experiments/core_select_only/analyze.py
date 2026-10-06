#!/usr/bin/env python3
"""core_select_only —— T0–T6 逐条实测 + 判定（三选一），写 results/summary.json。

判据逐字来自 PREREG §3（跑前写死，跑后未改）。
用法：uv run python experiments/core_select_only/analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "select_semantic_joint"))

from common import BAND, OLD_CARDS, SE_1250, TWO_SE  # noqa: E402  (只读 import)

RESULTS = HERE / "results"
SEEDS = (42, 43)
ARMS = ("joint", "selonly")
#: 冻结核基线（select_pool 臂 A = select_semantic_joint `frozen`，逐项 Δ=0.00 已验）
FROZEN = {42: {"heldout_pair": 49.52, "shifted_pos": 92.56, "test": 94.48},
          43: {"heldout_pair": 51.04, "shifted_pos": 92.48, "test": 94.48}}
#: T0 门禁：select_semantic_joint 主臂（必须零误差复现）
JOINT_EXPECT = {42: {"heldout_pair": 59.36, "shifted_pos": 99.60, "test": 99.88},
                43: {"heldout_pair": 63.20, "shifted_pos": 99.60, "test": 99.84}}


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"缺结果：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def acc(r: dict, key: str) -> float:
    return round(r["eval"]["adv_by_ctype"][key]["acc"] * 100.0, 2)


def split_acc(r: dict, s: str) -> float:
    return round(r["eval"][s]["acc"] * 100.0, 2)


def main() -> int:
    se = round(SE_1250 * 100, 4)
    two_se = round(TWO_SE * 100, 4)
    out: dict = {"se_pt": se, "two_se_pt": two_se, "runs": {}, "criteria": {}}

    real = {(a, s): load(f"{a}_s{s}") for a in ARMS for s in SEEDS}
    rnd = {s: load(f"selonly_s{s}_rand") for s in SEEDS}
    for k, v in {**real, **rnd}.items():
        out["runs"][v["name"]] = {"heldout": acc(v, "heldout_pair"),
                                  "shifted": acc(v, "shifted_pos"),
                                  "test": split_acc(v, "test"),
                                  "train": split_acc(v, "train"),
                                  "core_drift": v["core_drift"], "timing": v["timing"],
                                  "mount": v["mount_evidence"]["old_heads_mounted_during_training"],
                                  "randlabel_n_diff": v["randlabel_n_diff"]}

    # ---- T0（门禁）：joint 零误差复现 select_semantic_joint 主臂 ----
    t0 = {}
    for s in SEEDS:
        r = real[("joint", s)]
        t0[str(s)] = {}
        for k in ("heldout_pair", "shifted_pos"):
            got = acc(r, k)
            exp = JOINT_EXPECT[s][k]
            d = round(got - exp, 2)
            t0[str(s)][k] = {"got": got, "expect": exp, "delta": d, "zero_error": d == 0.0}
        g = split_acc(r, "test")
        t0[str(s)]["test"] = {"got": g, "expect": JOINT_EXPECT[s]["test"],
                              "delta": round(g - JOINT_EXPECT[s]["test"], 2),
                              "zero_error": round(g - JOINT_EXPECT[s]["test"], 2) == 0.0,
                              "report_only": True}
    t0_pass = all(v["zero_error"] for sd in t0.values()
                  for k, v in sd.items() if not v.get("report_only"))
    out["criteria"]["T0"] = {"pass": t0_pass, "detail": t0,
                             "note": "门槛 = heldout/shifted 逐 seed Δ = 0.00（零误差）"}
    print(f"T0 joint 零误差复现：{'过 ✅' if t0_pass else '不过 ❌'} "
          f"{json.dumps(t0, ensure_ascii=False)}")

    # ---- T1（主）：selonly heldout vs 冻结核基线 ----
    t1 = {}
    t1_pass = True
    for s in SEEDS:
        got = acc(real[("selonly", s)], "heldout_pair")
        base = FROZEN[s]["heldout_pair"]
        d = round(got - base, 2)
        ok = d > two_se
        t1[f"s{s}"] = {"selonly": got, "frozen": base, "delta_pt": d, "two_se": two_se,
                       "delta_over_se": round(d / se, 3), "margin_over_2se_pt": round(d - two_se, 2),
                       "pass": bool(ok)}
        t1_pass = t1_pass and ok
    out["criteria"]["T1"] = {"pass": t1_pass, "detail": t1,
                             "note": "Δ > 2×SE=2.828pt 且两 seed 同号（都过）"}
    print(f"T1 主判据：{'过 ✅' if t1_pass else '不过 ❌'} {json.dumps(t1, ensure_ascii=False)}")

    # ---- T2：selonly vs joint（老任务是否必需） ----
    t2 = {}
    t2_within = True
    for s in SEEDS:
        a_ = acc(real[("selonly", s)], "heldout_pair")
        j_ = acc(real[("joint", s)], "heldout_pair")
        d = round(a_ - j_, 2)
        ok = abs(d) <= two_se
        t2[f"s{s}"] = {"selonly": a_, "joint": j_, "delta_pt": d, "se_pt": se,
                       "delta_over_se": round(d / se, 3), "within_2se": bool(ok)}
        t2_within = t2_within and ok
    out["criteria"]["T2"] = {"pass": t2_within, "detail": t2,
                             "note": "报差值与 SE；噪声内 = |Δ| ≤ 2.828pt（逐 seed）"}
    print(f"T2 selonly vs joint：{'噪声内 ✅' if t2_within else '超出噪声 ❌'} "
          f"{json.dumps(t2, ensure_ascii=False)}")

    # ---- T3：selonly shifted 不塌 ----
    t3 = {}
    t3_pass = True
    for s in SEEDS:
        got = acc(real[("selonly", s)], "shifted_pos")
        base = FROZEN[s]["shifted_pos"]
        thr = round(base - two_se, 2)
        ok = got >= thr
        t3[f"s{s}"] = {"selonly": got, "frozen": base, "threshold": thr, "delta_pt": round(got - base, 2),
                       "pass": bool(ok)}
        t3_pass = t3_pass and ok
    out["criteria"]["T3"] = {"pass": t3_pass, "detail": t3,
                             "note": "selonly shifted ≥ 冻结基线 − 2.828pt（逐 seed）"}
    print(f"T3 shifted 不塌：{'过 ✅' if t3_pass else '不过 ❌'} {json.dumps(t3, ensure_ascii=False)}")

    # ---- T4（门禁）：selonly 随机标签对照 ----
    t4 = {}
    t4_pass = True
    for s in SEEDS:
        r = rnd[s]
        row = {k: acc(r, k) for k in ("heldout_pair", "shifted_pos")}
        ok = all(v <= round(50 + two_se, 2) for v in row.values())
        executed = bool(r["randlabel_n_diff"] and r["randlabel_n_diff"] > 0)
        t4[f"selonly_s{s}"] = dict(row, threshold=round(50 + two_se, 2),
                                   **{"pass": bool(ok and executed)},
                                   randlabel_n_diff=r["randlabel_n_diff"],
                                   train_true_labels=split_acc(r, "train"),
                                   executed=executed)
        t4_pass = t4_pass and ok and executed
    out["criteria"]["T4"] = {"pass": t4_pass, "detail": t4,
                             "note": "selonly 随机标签：heldout 与 shifted 均 ≤ 52.83，且打乱真执行"}
    print(f"T4 随机标签门禁：{'过 ✅' if t4_pass else '不过 ❌'} {json.dumps(t4, ensure_ascii=False)}")

    # ---- T5（代价，必报）：selonly 四张老卡 Δ ----
    t5 = {}
    for a in ARMS:
        for s in SEEDS:
            for n in OLD_CARDS:
                v = real[(a, s)]["r5"][n]
                t5[f"{a}_{n}_s{s}"] = {"step0": round(v["step0"], 6), "post": round(v["post"], 6),
                                       "delta": round(v["delta"], 6), "band": BAND[n],
                                       "degraded_beyond_band": bool(v["delta"] < -BAND[n])}
    t5_reported = len(t5) == 16
    out["criteria"]["T5"] = {"pass": t5_reported, "detail": t5,
                             "note": "必报：逐卡 Δ 与各自带；预期 selonly 会退化（不是门禁）"}
    print(f"T5 老卡代价（必报 16 项齐全={t5_reported}）：{json.dumps(t5, ensure_ascii=False)}")

    # ---- T6：核漂移 / 步时 / 峰值显存 ----
    t6 = {}
    for a in ARMS:
        for s in SEEDS:
            r = real[(a, s)]
            t6[f"{a}_s{s}"] = {"core_drift": r["core_drift"], "timing": r["timing"],
                               "freeze": r["freeze"],
                               "old_heads_mounted_during_training":
                                   r["mount_evidence"]["old_heads_mounted_during_training"],
                               "old_forward_count": r["mount_evidence"]["old_forward_count"]}
            if r["randlabel"]:
                pass
    for s in SEEDS:
        r = rnd[s]
        t6[f"selonly_rand_s{s}"] = {"core_drift": r["core_drift"], "timing": r["timing"]}
    out["criteria"]["T6"] = {"pass": True, "detail": t6,
                             "note": "必报：漂移 + 稳态步时 + 峰值显存"}
    print("T6 " + json.dumps({k: {"drift": v["core_drift"],
                                  "sec/step": v["timing"]["sec_per_step_steady"],
                                  "peak_mb": v["timing"]["peak_mem_mb"]}
                              for k, v in t6.items()}, ensure_ascii=False))

    # ---- 判定（三选一，PREREG §3 跑前写死） ----
    sel_h = {s: acc(real[("selonly", s)], "heldout_pair") for s in SEEDS}
    near50 = all(abs(sel_h[s] - 50) <= two_se for s in SEEDS)
    gates_ok = t0_pass and t4_pass
    if not gates_ok:
        verdict = "③ 证据不足"
        why = "门禁不过：" + "、".join(k for k, v in (("T0", t0_pass), ("T4", t4_pass)) if not v)
    elif t1_pass and t2_within:
        verdict = "① 增益来自 select 梯度进核"
        why = "T1 过（两 seed Δ > 2×SE）且 T2 过（与 joint 差距 ≤ 2×SE）⇒ 老任务锚定非必需"
    elif (not t1_pass) and near50:
        verdict = "② 老任务锚定是必需的"
        why = f"T1 不过 且 selonly heldout ≈50%（{sel_h}）⇒ 只给 select 梯度不产生增益"
    elif t1_pass and not t2_within:
        verdict = "③ 证据不足"
        why = ("T1 过但 T2 不过：select 梯度有增益，但 selonly 低于 joint "
               "⇒ 两机制都有份（不满足①的『老任务非必需』，也不满足②的『回到 50%』）")
    else:
        verdict = "③ 证据不足"
        why = f"T1 不过 且 selonly 也不 ≈50%（{sel_h}）"
    out["verdict"] = {"verdict": verdict, "why": why,
                      "gates": {"T0": t0_pass, "T4": t4_pass},
                      "T1": t1_pass, "T2": t2_within, "T3": t3_pass, "T5": t5_reported}
    print(f"\n判定 = {verdict}（{why}）")

    (RESULTS / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("[save] results/summary.json")
    print("ANALYZE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
