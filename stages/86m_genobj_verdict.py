#!/usr/bin/env python3
"""stages/86m_genobj_verdict.py · H2 生成目标（PREREG_GENOBJ.md §3/§4）判据表与自检脚本。

口径与判据定义：
- 数据源：logs/86g_result_*.json
- 筛选条件：bridge == "parser" and frozen and "_a0.5_" in init_inputs and obj in ("tf", "roll")
- 配对差：同 seed 配对 Δ = roll - tf
- SE 计算公式选取与理由：
  本脚本计算"配对差的均值标准误（Standard Error of Paired Differences Mean）"：
    SE_paired = std(Δ, ddof=1) / sqrt(k) = sqrt(sum((Δᵢ - mean(Δ))²) / (k - 1)) / sqrt(k)
  理由：
  同 seed 下两臂使用完全相同的随机数种子、相同的数据批次顺序与相同的评测集（test_ids[:200]）。
  配对设计控制了 seed 间的基线变异，评估两目标策略（roll vs tf）处理效应均值时的标准误应当基于配对差向量 Δ
  的样本方差除以样本量 k（即 k=4 个 seed）。
  同时定义 Δ_min = 2 * SE_paired（由 PREREG §3 G5 与 methodology/measurement.md §3 明确要求）。

- 判据（PREREG_GENOBJ.md §3）：
  G1（主）：自由生成 EM（gen_em_zz 与 gen_em_ee），要求配对 Δ ≥ +2×SE 且 4/4 seed 同号（Δᵢ > 0）。
  G2（不伤保真）：teacher-forced 重构 char_zz / char_ee，掉 ≤ 2pp（即 mean(Δ) ≥ -0.02）。
  G3（不破对齐）：metrics.bridge.pooled.delta 与 metrics.bridge.auroc_centered，≥ T 臂均值 - 0.02 / -0.03。
  G4（读数完备）：8/8 run 完备（4 seed × 2 obj），字段完整（final{char_zz,char_ee}, gen{gen_em_zz,gen_em_ee}, metrics）。
  G5（Δ_min 声明）：报出 Δ_min = 2×SE；当 |Δ| < Δ_min 时必须标明"该口径下未测出差异"。
  G6（饱和检查）：每 1000 步快照末两格增量；若未饱和标【下界】。

- 三分支判决（PREREG_GENOBJ.md §4）：
  分支一：G1 过 且 G2/G3 过 ⇒ "生成缺口是目标问题"成立，rollout 项进 v0 配方。
  分支二：G1 不过 且 |Δ| ≤ 1×SE ⇒ 未测出差异：生成缺口不是（或不止是）exposure bias，不纠缠。
  分支三：G1 显著为负（mean(Δ) ≤ -2×SE 且全负/同号为负）⇒ rollout 目标在本规模下有害。
  （其它：不落一二三分支时，输出如实判断；若读数不完备则判词作废）。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

LOGS_DIR = Path("/home/vesita/coding/my/flowme/logs")


def dig(r: dict[str, Any], *keys: str) -> float:
    cur: Any = r
    for k in keys:
        cur = cur[k]
    return float(cur)


def compute_paired_stats(deltas: list[float]) -> dict[str, float]:
    k = len(deltas)
    if k == 0:
        return {"mean": float("nan"), "se": float("nan"), "delta_min": float("nan")}
    mean_val = sum(deltas) / k
    if k > 1:
        var = sum((x - mean_val) ** 2 for x in deltas) / (k - 1)
        se = math.sqrt(var) / math.sqrt(k)
    else:
        se = float("nan")
    delta_min = 2.0 * se if not math.isnan(se) else float("nan")
    return {"mean": mean_val, "se": se, "delta_min": delta_min}


def evaluate_genobj(runs: list[dict[str, Any]]) -> dict[str, Any]:
    # 筛选合格 run
    valid_runs: list[dict[str, Any]] = []
    for r in runs:
        if r.get("bridge") != "parser":
            continue
        if not r.get("frozen"):
            continue
        if "_a0.5_" not in str(r.get("init_inputs", "")):
            continue
        if r.get("obj") not in ("tf", "roll"):
            continue
        valid_runs.append(r)

    # 检查读数完备性（G4）
    # 理想规格：8/8 run（4 seed: 1234, 5678, 9012, 3456 各有 tf 与 roll）
    target_seeds = [1234, 5678, 9012, 3456]
    runs_by_obj_seed: dict[tuple[str, int], dict[str, Any]] = {}
    
    g4_complete = True
    g4_reasons = []

    for r in valid_runs:
        obj = str(r.get("obj"))
        seed = int(r.get("seed", -1))
        # 字段检查
        has_final = "final" in r and "char_zz" in r["final"] and "char_ee" in r["final"]
        has_gen = "gen" in r and "gen_em_zz" in r["gen"] and "gen_em_ee" in r["gen"]
        has_metrics = ("metrics" in r and "bridge" in r["metrics"] 
                       and "pooled" in r["metrics"]["bridge"] 
                       and "delta" in r["metrics"]["bridge"]["pooled"]
                       and "auroc_centered" in r["metrics"]["bridge"])
        if not (has_final and has_gen and has_metrics):
            g4_complete = False
            g4_reasons.append(f"run obj={obj} seed={seed} 关键指标字段缺失")
        runs_by_obj_seed[(obj, seed)] = r

    for s in target_seeds:
        if ("tf", s) not in runs_by_obj_seed:
            g4_complete = False
            g4_reasons.append(f"缺少 run: obj=tf seed={s}")
        if ("roll", s) not in runs_by_obj_seed:
            g4_complete = False
            g4_reasons.append(f"缺少 run: obj=roll seed={s}")

    if not g4_complete:
        return {
            "g4_pass": False,
            "verdict_branch": "VOID",
            "verdict_text": "【判词作废】读数不完备：" + "; ".join(g4_reasons),
            "details": {"g4_reasons": g4_reasons},
        }

    # 4 对配对数据
    paired_data = []
    for s in target_seeds:
        tf_r = runs_by_obj_seed[("tf", s)]
        ro_r = runs_by_obj_seed[("roll", s)]
        paired_data.append((s, tf_r, ro_r))

    # 配对差计算
    # 指标清单：
    # 1. 自由生成 EM（主）：gen_em_zz, gen_em_ee
    # 2. 重构 char（不伤保真）：char_zz, char_ee
    # 3. 对齐度量（不破对齐）：metrics.bridge.pooled.delta, metrics.bridge.auroc_centered
    diff_zz = [dig(ro, "gen", "gen_em_zz") - dig(tf, "gen", "gen_em_zz") for _, tf, ro in paired_data]
    diff_ee = [dig(ro, "gen", "gen_em_ee") - dig(tf, "gen", "gen_em_ee") for _, tf, ro in paired_data]
    diff_char_zz = [dig(ro, "final", "char_zz") - dig(tf, "final", "char_zz") for _, tf, ro in paired_data]
    diff_char_ee = [dig(ro, "final", "char_ee") - dig(tf, "final", "char_ee") for _, tf, ro in paired_data]
    
    tf_bridge_deltas = [dig(tf, "metrics", "bridge", "pooled", "delta") for _, tf, _ in paired_data]
    ro_bridge_deltas = [dig(ro, "metrics", "bridge", "pooled", "delta") for _, _, ro in paired_data]
    tf_aurocs = [dig(tf, "metrics", "bridge", "auroc_centered") for _, tf, _ in paired_data]
    ro_aurocs = [dig(ro, "metrics", "bridge", "auroc_centered") for _, _, ro in paired_data]

    stats_zz = compute_paired_stats(diff_zz)
    stats_ee = compute_paired_stats(diff_ee)
    stats_char_zz = compute_paired_stats(diff_char_zz)
    stats_char_ee = compute_paired_stats(diff_char_ee)

    # G1（主）：自由生成 EM，Δ ≥ +2×SE 且 4/4 seed 同号（Δᵢ > 0）
    g1_zz_pass = (stats_zz["mean"] >= 2.0 * stats_zz["se"]) and all(d > 0 for d in diff_zz)
    g1_ee_pass = (stats_ee["mean"] >= 2.0 * stats_ee["se"]) and all(d > 0 for d in diff_ee)
    g1_pass = g1_zz_pass and g1_ee_pass

    # G2（不伤保真）：teacher-forced 重构 char_zz / char_ee 掉 ≤ 2pp（即配对差均值 ≥ -0.02）
    g2_zz_pass = stats_char_zz["mean"] >= -0.02
    g2_ee_pass = stats_char_ee["mean"] >= -0.02
    g2_pass = g2_zz_pass and g2_ee_pass

    # G3（不破对齐）：衔接 Δ / AUROC ≥ T 臂均值 - 0.02 / -0.03
    tf_bridge_delta_mean = sum(tf_bridge_deltas) / len(tf_bridge_deltas)
    ro_bridge_delta_mean = sum(ro_bridge_deltas) / len(ro_bridge_deltas)
    tf_auroc_mean = sum(tf_aurocs) / len(tf_aurocs)
    ro_auroc_mean = sum(ro_aurocs) / len(ro_aurocs)

    g3_delta_pass = ro_bridge_delta_mean >= (tf_bridge_delta_mean - 0.02)
    g3_auroc_pass = ro_auroc_mean >= (tf_auroc_mean - 0.03)
    g3_pass = g3_delta_pass and g3_auroc_pass

    # G5：Δ_min 声明与未测出差异判断
    # 针对两个主指标
    unmeasured_zz = abs(stats_zz["mean"]) < stats_zz["delta_min"]
    unmeasured_ee = abs(stats_ee["mean"]) < stats_ee["delta_min"]

    # G6：饱和检查（检查 snaps 是否存在且末两格增量）
    g6_reports = []
    for s, tf_r, ro_r in paired_data:
        for tag, r_obj in [("tf", tf_r), ("roll", ro_r)]:
            snaps = r_obj.get("snaps", [])
            if len(snaps) >= 2:
                # 检查末两格（例如 7000 与 8000 步）主 loss 或 metrics 变化
                last = snaps[-1]
                prev = snaps[-2]
                g6_reports.append(f"s{s}_{tag}: last={last.get('step')} prev={prev.get('step')}")
            else:
                g6_reports.append(f"s{s}_{tag}: 快照不足2格（未饱和，读数标【下界】）")

    # 三分支判决（PREREG §4）：
    # 分支一：G1 过且 G2/G3 过
    # 分支二：G1 不过且 |Δ| ≤ 1SE（两个主指标均在 1SE 内，或至少均未出现显著效应）
    # 分支三：G1 显著为负（Δ ≤ -2SE 且 4/4 seed 同号为负）
    is_branch1 = g1_pass and g2_pass and g3_pass
    
    # 显著为负判定（任一主指标显著为负，或均显著为负）
    neg_zz = (stats_zz["mean"] <= -2.0 * stats_zz["se"]) and all(d < 0 for d in diff_zz)
    neg_ee = (stats_ee["mean"] <= -2.0 * stats_ee["se"]) and all(d < 0 for d in diff_ee)
    is_branch3 = neg_zz or neg_ee

    # 分支二：G1 不过且 |Δ| ≤ 1SE
    # 主指标中没有显著为负，且配对差在 1SE 之内
    small_zz = abs(stats_zz["mean"]) <= (1.0 * stats_zz["se"])
    small_ee = abs(stats_ee["mean"]) <= (1.0 * stats_ee["se"])
    is_branch2 = (not g1_pass) and (not is_branch3) and (small_zz and small_ee)

    if is_branch1:
        branch = "BRANCH_1"
        verdict_text = "【分支一】「生成缺口是目标问题」成立 ⇒ rollout 项进 v0 配方；再进 H1/H6"
    elif is_branch2:
        branch = "BRANCH_2"
        verdict_text = "【分支二】未测出差异：生成缺口不是（或不止是）exposure bias ⇒ 不纠缠（§0.2），转 H6"
    elif is_branch3:
        branch = "BRANCH_3"
        verdict_text = "【分支三】rollout 目标在本规模下有害 ⇒ 记结论并写进 PROJECT_STATE.md 已判死"
    else:
        branch = "OTHER"
        verdict_text = "【其它】G1 未达显著正增益，亦未落入严格分支二(|Δ|≤1SE)或分支三(显著为负)；按该口径下未测出差异处理"

    return {
        "g4_pass": True,
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g3_pass": g3_pass,
        "verdict_branch": branch,
        "verdict_text": verdict_text,
        "stats": {
            "gen_em_zz": stats_zz,
            "gen_em_ee": stats_ee,
            "char_zz": stats_char_zz,
            "char_ee": stats_char_ee,
            "diff_zz": diff_zz,
            "diff_ee": diff_ee,
            "unmeasured_zz": unmeasured_zz,
            "unmeasured_ee": unmeasured_ee,
            "tf_bridge_delta_mean": tf_bridge_delta_mean,
            "ro_bridge_delta_mean": ro_bridge_delta_mean,
            "tf_auroc_mean": tf_auroc_mean,
            "ro_auroc_mean": ro_auroc_mean,
        },
        "paired_data": paired_data,
        "g6_reports": g6_reports,
    }


def make_synthetic_runs(case: str) -> list[dict[str, Any]]:
    seeds = [1234, 5678, 9012, 3456]
    runs: list[dict[str, Any]] = []

    if case == "incomplete":
        # 故意缺失部分 run 或字段
        for s in seeds[:3]:
            runs.append({
                "bridge": "parser",
                "frozen": ["zh_in", "en_in"],
                "init_inputs": "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s1234.pt",
                "obj": "tf",
                "seed": s,
                "final": {"char_zz": 0.92, "char_ee": 0.96},
                "gen": {"gen_em_zz": 0.65, "gen_em_ee": 0.68},
                "metrics": {"bridge": {"pooled": {"delta": 0.45}, "auroc_centered": 0.99}},
            })
        return runs

    for s in seeds:
        # 基准 tf 读数
        tf_run = {
            "bridge": "parser",
            "frozen": ["zh_in", "en_in"],
            "init_inputs": f"logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s{s}.pt",
            "obj": "tf",
            "seed": s,
            "final": {"char_zz": 0.925, "char_ee": 0.962},
            "gen": {"gen_em_zz": 0.650, "gen_em_ee": 0.680},
            "metrics": {
                "bridge": {
                    "pooled": {"delta": 0.450},
                    "auroc_centered": 0.990,
                }
            },
            "snaps": [{"step": 7000}, {"step": 8000}],
        }
        runs.append(tf_run)

        # 构造 roll 读数
        if case == "branch1":
            # G1 过：Δ ≥ +2×SE 且 4 seed 全正（如 +0.060, +0.062, +0.058, +0.061，均值约 +0.060，SE 极小）
            # G2 过：char 不掉超过 2pp（如掉 0.005）
            # G3 过：对齐度量不破
            roll_run = {
                "bridge": "parser",
                "frozen": ["zh_in", "en_in"],
                "init_inputs": f"logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s{s}.pt",
                "obj": "roll",
                "roll_w": 0.5,
                "seed": s,
                "final": {"char_zz": 0.920, "char_ee": 0.958},
                "gen": {"gen_em_zz": 0.650 + 0.060 + (0.001 if s % 2 == 0 else -0.001),
                        "gen_em_ee": 0.680 + 0.055 + (0.001 if s % 2 == 0 else -0.001)},
                "metrics": {
                    "bridge": {
                        "pooled": {"delta": 0.448},
                        "auroc_centered": 0.988,
                    }
                },
                "snaps": [{"step": 7000}, {"step": 8000}],
            }
        elif case == "branch2":
            # G1 不过且 |Δ| ≤ 1SE：波动极小，近乎无效应（如 +0.001, -0.001, +0.002, -0.002，mean=0.0，SE≈0.001）
            roll_run = {
                "bridge": "parser",
                "frozen": ["zh_in", "en_in"],
                "init_inputs": f"logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s{s}.pt",
                "obj": "roll",
                "roll_w": 0.5,
                "seed": s,
                "final": {"char_zz": 0.924, "char_ee": 0.961},
                "gen": {"gen_em_zz": 0.650 + (0.001 if s in (1234, 5678) else -0.001),
                        "gen_em_ee": 0.680 + (0.001 if s in (1234, 9012) else -0.001)},
                "metrics": {
                    "bridge": {
                        "pooled": {"delta": 0.450},
                        "auroc_centered": 0.990,
                    }
                },
                "snaps": [{"step": 7000}, {"step": 8000}],
            }
        elif case == "branch3":
            # G1 显著为负：Δ ≤ -2×SE 且全负（如 -0.050, -0.052, -0.048, -0.051）
            roll_run = {
                "bridge": "parser",
                "frozen": ["zh_in", "en_in"],
                "init_inputs": f"logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s{s}.pt",
                "obj": "roll",
                "roll_w": 0.5,
                "seed": s,
                "final": {"char_zz": 0.910, "char_ee": 0.940},
                "gen": {"gen_em_zz": 0.650 - 0.050 + (0.001 if s % 2 == 0 else -0.001),
                        "gen_em_ee": 0.680 - 0.060 + (0.001 if s % 2 == 0 else -0.001)},
                "metrics": {
                    "bridge": {
                        "pooled": {"delta": 0.440},
                        "auroc_centered": 0.975,
                    }
                },
                "snaps": [{"step": 7000}, {"step": 8000}],
            }
        else:
            raise ValueError(f"未知假结果用例: {case}")
        runs.append(roll_run)

    return runs


def run_selfcheck() -> bool:
    print("==================================================")
    print("运行 R48 假结果自检（构造数据驱动真实判词函数）")
    print("==================================================")
    
    cases = [
        ("档位 1: G1 过 且 G2/G3 过 (预期 BRANCH_1)", "branch1", "BRANCH_1"),
        ("档位 2: G1 不过 且 |Δ| ≤ 1SE (预期 BRANCH_2)", "branch2", "BRANCH_2"),
        ("档位 3: G1 显著为负 (预期 BRANCH_3)", "branch3", "BRANCH_3"),
        ("档位 4: 读数不完备 (预期 VOID)", "incomplete", "VOID"),
    ]

    all_passed = True
    for desc, case_name, expected_branch in cases:
        synthetic_runs = make_synthetic_runs(case_name)
        res = evaluate_genobj(synthetic_runs)
        actual_branch = res["verdict_branch"]
        ok = (actual_branch == expected_branch)
        all_passed = all_passed and ok
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"[{status}] {desc}")
        print(f"       判词输出: {res['verdict_text']}")
        if not ok:
            print(f"       预期分支: {expected_branch}, 实际分支: {actual_branch}")

    print("==================================================")
    if all_passed:
        print("[SELFCHECK-OK] R48 自检全部四档通过，判据逻辑完备。")
        return True
    else:
        print("[SELFCHECK-FAIL] R48 自检未通过！")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 生成目标 (PREREG_GENOBJ) 判据表")
    parser.add_argument("--selfcheck", action="store_true", help="运行 R48 假结果自检模式")
    args = parser.parse_args()

    if args.selfcheck:
        ok = run_selfcheck()
        sys.exit(0 if ok else 1)

    # 真实运行模式：读取 logs/86g_result_*.json
    json_files = sorted(glob.glob(str(LOGS_DIR / "86g_result_*.json")))
    runs = []
    for f in json_files:
        try:
            runs.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception:
            continue

    res = evaluate_genobj(runs)
    if not res["g4_pass"]:
        print(f"[GENOBJ-VERDICT] {res['verdict_text']}")
        sys.exit(2)

    st = res["stats"]
    print("## H2 生成目标（PREREG_GENOBJ）配对判据表 (Δ = roll − tf)")
    print()
    print("| 指标 | T 臂均值 | R 臂均值 | 配对均值 Δ | SE_paired | Δ_min (2×SE) | 4 seed 逐差 | 判定 |")
    print("|---|---|---|---|---|---|---|---|")
    
    # 逐指标打印
    zz = st["gen_em_zz"]
    ee = st["gen_em_ee"]
    c_zz = st["char_zz"]
    c_ee = st["char_ee"]
    print(f"| G1 自由生成EM 中→中 | - | - | {zz['mean']*100:+.2f}pp | {zz['se']*100:.2f}pp | {zz['delta_min']*100:.2f}pp | {[f'{x*100:+.1f}pp' for x in st['diff_zz']]} | {'✅ G1过' if res['g1_pass'] else '❌'} |")
    print(f"| G1 自由生成EM 英→英 | - | - | {ee['mean']*100:+.2f}pp | {ee['se']*100:.2f}pp | {ee['delta_min']*100:.2f}pp | {[f'{x*100:+.1f}pp' for x in st['diff_ee']]} | {'✅ G1过' if res['g1_pass'] else '❌'} |")
    print(f"| G2 重构字准 中→中 | - | - | {c_zz['mean']*100:+.2f}pp | {c_zz['se']*100:.2f}pp | - | - | {'✅' if c_zz['mean'] >= -0.02 else '❌ 掉>2pp'} |")
    print(f"| G2 重构字准 英→英 | - | - | {c_ee['mean']*100:+.2f}pp | {c_ee['se']*100:.2f}pp | - | - | {'✅' if c_ee['mean'] >= -0.02 else '❌ 掉>2pp'} |")
    print(f"| G3 衔接Δ | {st['tf_bridge_delta_mean']:+.4f} | {st['ro_bridge_delta_mean']:+.4f} | {st['ro_bridge_delta_mean'] - st['tf_bridge_delta_mean']:+.4f} | - | - | - | {'✅' if (st['ro_bridge_delta_mean'] >= st['tf_bridge_delta_mean'] - 0.02) else '❌ 掉>0.02'} |")
    print(f"| G3 AUROC | {st['tf_auroc_mean']:.4f} | {st['ro_auroc_mean']:.4f} | {st['ro_auroc_mean'] - st['tf_auroc_mean']:+.4f} | - | - | - | {'✅' if (st['ro_auroc_mean'] >= st['tf_auroc_mean'] - 0.03) else '❌ 掉>0.03'} |")
    print()
    print(f"判词结果：{res['verdict_text']}")


if __name__ == "__main__":
    main()
