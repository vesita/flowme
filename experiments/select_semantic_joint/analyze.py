#!/usr/bin/env python3
"""select_semantic_joint —— R0–R6 逐条实测 + 判定（三选一），写 results/summary.json。

判据逐字来自 PREREG §3（跑前写死，跑后未改）。
用法：uv run python experiments/select_semantic_joint/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import BAND, OLD_CARDS, RESULTS, R0_EXPECT, SE_1250, TWO_SE  # noqa: E402

ARM = ("frozen", "joint")
SEEDS = (42, 43)


def load(arm: str, seed: int, rand: bool = False) -> dict:
    n = f"{arm}_s{seed}{'_rand' if rand else ''}"
    p = RESULTS / f"{n}.json"
    if not p.exists():
        raise SystemExit(f"缺结果：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    out: dict = {"se": SE_1250, "two_se": TWO_SE, "runs": {}, "criteria": {}}

    real = {(a, s): load(a, s) for a in ARM for s in SEEDS}
    rnd = {(a, s): load(a, s, True) for a in ARM for s in SEEDS}

    def acc(r: dict, key: str) -> float:
        return r["eval"]["adv_by_ctype"][key]["acc"] * 100.0

    # ---- R0：核对臂零误差复现 select_pool 臂 A ----
    r0 = {}
    for s in SEEDS:
        f = real[("frozen", s)]
        r0[str(s)] = {}
        for k in ("heldout_pair", "shifted_pos"):
            got = acc(f, k)
            exp = R0_EXPECT[s][k]
            d = round(got - exp, 6)
            r0[str(s)][k] = {"got": round(got, 4), "expect": exp, "delta": d,
                             "within_2se": abs(d) <= TWO_SE, "zero_error": d == 0.0}
    r0_pass = all(v["zero_error"] for sd in r0.values() for v in sd.values())
    out["criteria"]["R0"] = {"pass": r0_pass, "detail": r0,
                             "note": "门槛=逐 seed Δ 必须为 0.00（零误差）"}
    print(f"R0 核对臂零误差复现：{'过 ✅' if r0_pass else '不过 ❌'} {json.dumps(r0, ensure_ascii=False)}")

    # ---- R4：step-0 逐位（硬门禁，先判） ----
    r4 = {}
    for s in SEEDS:
        base = json.loads((RESULTS / f"r4_baseline_s{s}.json").read_text(encoding="utf-8"))
        step0 = real[("joint", s)]["r4_step0"]
        r4[str(s)] = {
            "baseline": base["combined_sha256"],
            "step0": step0["combined_sha256"],
            "equal": base["combined_sha256"] == step0["combined_sha256"],
            "per_card_equal": {n: base[n]["sha256"] == step0[n]["sha256"] for n in OLD_CARDS},
            "baseline_exact": {n: base[n]["exact"] for n in OLD_CARDS},
            "step0_exact": {n: step0[n]["exact"] for n in OLD_CARDS},
            "delta_vs_capmap": {n: base[n]["delta_vs_capmap"] for n in OLD_CARDS},
        }
    r4_pass = all(r4[str(s)]["equal"] and all(r4[str(s)]["per_card_equal"].values())
                  for s in SEEDS)
    out["criteria"]["R4"] = {"pass": r4_pass, "detail": r4,
                             "note": "step-0 老卡逐样本 pred/exact 的 sha256 == 不接本任务配置"}
    print(f"R4 step-0 逐位不变：{'过 ✅' if r4_pass else '不过 ❌'} "
          f"{ {s: r4[str(s)]['equal'] for s in SEEDS} }")

    # ---- R3：随机标签对照 ----
    r3 = {}
    r3_pass = True
    for a in ARM:
        for s in SEEDS:
            r = rnd[(a, s)]
            row = {k: round(acc(r, k), 4) for k in ("heldout_pair", "shifted_pos")}
            ok = all(v <= 50 + TWO_SE * 100 for v in row.values())
            r3[f"{a}_s{s}"] = dict(row, pass_=ok,
                                   train=round(r["eval"]["train"]["acc"] * 100, 2),
                                   test=round(r["eval"]["test"]["acc"] * 100, 2),
                                   n_diff_executed=r["randlabel_n_diff"])
            r3_pass = r3_pass and ok
    out["criteria"]["R3"] = {"pass": r3_pass, "detail": r3,
                             "note": "4 跑（两臂 × 两 seed）heldout 与 shifted 均 ≤ 52.83"}
    print(f"R3 随机标签对照：{'过 ✅' if r3_pass else '不过 ❌'} {json.dumps(r3, ensure_ascii=False)}")

    # ---- R1（主）/ R2（副） ----
    r1, r2 = {}, {}
    r1_pass = r2_pass = True
    same_sign = []
    for s in SEEDS:
        fj, ff = real[("joint", s)], real[("frozen", s)]
        hj, hf = acc(fj, "heldout_pair"), acc(ff, "heldout_pair")
        sj, sf = acc(fj, "shifted_pos"), acc(ff, "shifted_pos")
        thr_pt = TWO_SE * 100
        d_h = round(hj - hf, 4)
        d_s = round(sj - sf, 4)
        thr = TWO_SE * 100                        # 2.8284pt
        same_sign.append(d_h > 0)
        ok_h = d_h > thr
        ok_s = sj >= sf - thr
        r1[f"s{s}"] = {"joint": round(hj, 4), "frozen": round(hf, 4),
                       "delta_pt": d_h, "se_pt": round(SE_1250 * 100, 4),
                       "delta_over_se": round(d_h / (SE_1250 * 100), 3),
                       "delta_over_se_diff": round(d_h / (SE_1250 * 100 * 1.4142), 3),
                       "margin_vs_2se_pt": round(d_h - TWO_SE * 100, 4),
                       "pass": bool(ok_h),
                       "joint_vs_50_over_se": round((hj - 50) / (SE_1250 * 100), 3)}
        r2[f"s{s}"] = {"joint": round(sj, 4), "frozen": round(sf, 4),
                       "delta_pt": d_s, "threshold": round(sf - thr_pt, 4),
                       "pass": bool(ok_s)}
        r1_pass = r1_pass and ok_h
        r2_pass = r2_pass and ok_s
    r1_pass = r1_pass and all(same_sign) and len(same_sign) == 2
    out["criteria"]["R1"] = {"pass": r1_pass, "detail": r1,
                             "note": "Δ > 2×SE=2.828pt 且两 seed 同号（都过）"}
    out["criteria"]["R2"] = {"pass": r2_pass, "detail": r2,
                             "note": "joint.shifted ≥ frozen.shifted − 2.828pt（逐 seed）"}
    print(f"R1 主判据：{'过 ✅' if r1_pass else '不过 ❌'} {json.dumps(r1, ensure_ascii=False)}")
    print(f"R2 副判据：{'过 ✅' if r2_pass else '不过 ❌'} {json.dumps(r2, ensure_ascii=False)}")

    # ---- R5：老卡不退化 ----
    r5, r5_pass = {}, True
    for s in SEEDS:
        for n in OLD_CARDS:
            v = real[("joint", s)]["r5"][n]
            ok = v["delta"] >= -BAND[n]
            r5[f"{n}_s{s}"] = {"step0": round(v["step0"], 6), "post": round(v["post"], 6),
                               "delta": round(v["delta"], 6), "band": BAND[n],
                               "pass": bool(ok)}
            r5_pass = r5_pass and ok
    out["criteria"]["R5"] = {"pass": r5_pass, "detail": r5,
                             "note": "报告项：Δ ≥ −各自噪声带（8 格）"}
    print(f"R5 老卡：{'过 ✅' if r5_pass else '不过 ❌'} {json.dumps(r5, ensure_ascii=False)}")

    # ---- R6：核漂移与代价 ----
    r6 = {}
    for s in SEEDS:
        for a in ARM:
            r = real[(a, s)]
            r6[f"{a}_s{s}"] = {"core_drift": r["core_drift"], "timing": r["timing"],
                               "freeze": r["freeze"],
                               "old_heads_frozen": bool(r["old_heads_freeze"]),
                               "recipe": r["recipe"]}
    out["criteria"]["R6"] = {"pass": True, "detail": r6, "note": "必报：漂移 + 步时 + 显存"}
    print("R6 代价 " + json.dumps({k: {"drift": v["core_drift"],
                                       "sec/step": v["timing"]["sec_per_step_steady"],
                                       "peak_mb": v["timing"]["peak_mem_mb"]}
                                   for k, v in r6.items()}, ensure_ascii=False))

    # ---- 判定（三选一，R3/R4 是门禁） ----
    gates_ok = r0_pass and r3_pass and r4_pass
    if not gates_ok:
        verdict = "③ 证据不足"
        why = "门禁不过：" + "、".join(k for k, v in
                                     (("R0", r0_pass), ("R3", r3_pass), ("R4", r4_pass))
                                     if not v)
    elif r1_pass:
        verdict = "② 墙是「核没学过这个任务」"
        why = "R1 过（两 seed 同号且 Δ > 2×SE）"
    elif r2_pass:
        verdict = "① 墙在表示"
        why = "R1 不过 且 R2 过"
    else:
        verdict = "③ 证据不足"
        why = "R1 不过且 R2 也不过（或只在一个 seed 过）"
    out["verdict"] = {"verdict": verdict, "why": why,
                      "gates": {"R0": r0_pass, "R3": r3_pass, "R4": r4_pass},
                      "R1": r1_pass, "R2": r2_pass, "R5": r5_pass}
    print(f"\n判定 = {verdict}（{why}）")

    (RESULTS / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("[save] results/summary.json")
    print("ANALYZE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
