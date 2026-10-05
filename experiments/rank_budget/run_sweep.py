#!/usr/bin/env python3
"""多随机种子复核验证：对 E1-E4 跑 3 个随机种子 (seed 42, 43, 44)，评估数字的稳定性与置信度。
生成最终统一的 results.json。
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "rank_budget"

SEEDS = [42, 43, 44]

def run_command(cmd: str):
    print(f">> {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout

def main():
    print("=== 开始多 seed 复核 (42, 43, 44) ===")
    all_results = {"seeds": SEEDS, "runs": {}}

    for seed in SEEDS:
        print(f"\n--- 运行 Seed {seed} ---")
        e1_out = EXP_DIR / f"e1_seed{seed}.json"
        e3_out = EXP_DIR / f"e3_seed{seed}.json"
        e4_out = EXP_DIR / f"e4_seed{seed}.json"

        run_command(f"uv run python experiments/rank_budget/measure_e1.py --seed {seed} --out {e1_out}")
        run_command(f"uv run python experiments/rank_budget/measure_e3.py --seed {seed} --out {e3_out}")
        run_command(f"uv run python experiments/rank_budget/measure_e4.py --seed {seed} --out {e4_out}")

        with open(e1_out, "r", encoding="utf-8") as f:
            e1_data = json.load(f)
        with open(e3_out, "r", encoding="utf-8") as f:
            e3_data = json.load(f)
        with open(e4_out, "r", encoding="utf-8") as f:
            e4_data = json.load(f)

        all_results["runs"][f"seed_{seed}"] = {
            "e1": e1_data,
            "e3": e3_data,
            "e4": e4_data,
        }

    # E2 是确定性的权重 SVD 分析，与数据采样 seed 无关
    e2_out = EXP_DIR / "e2_results.json"
    run_command(f"uv run python experiments/rank_budget/measure_e2.py --out {e2_out}")
    with open(e2_out, "r", encoding="utf-8") as f:
        e2_data = json.load(f)
    all_results["e2_deterministic"] = e2_data

    # 聚合核心指标统计
    summary = {
        "e1_combined_output_util": [all_results["runs"][f"seed_{s}"]["e1"]["combined"]["output"]["util"] for s in SEEDS],
        "e1_combined_output_erank": [all_results["runs"][f"seed_{s}"]["e1"]["combined"]["output"]["erank"] for s in SEEDS],
        "e3_sentiment_relation_cos": [all_results["runs"][f"seed_{s}"]["e3"]["cosine_similarity"]["sentiment"]["relation"] for s in SEEDS],
        "e3_sentiment_relation_min_angle": [all_results["runs"][f"seed_{s}"]["e3"]["principal_angles"]["relation"]["sentiment"]["min_angle_deg"] for s in SEEDS],
        "e4_4class_probe_acc": [all_results["runs"][f"seed_{s}"]["e4"]["four_classes"]["real_probe"]["accuracy"] for s in SEEDS],
        "e4_4class_probe_f1": [all_results["runs"][f"seed_{s}"]["e4"]["four_classes"]["real_probe"]["macro_f1"] for s in SEEDS],
        "e4_5class_probe_acc": [all_results["runs"][f"seed_{s}"]["e4"]["five_classes_with_neutral"]["real_probe"]["accuracy"] for s in SEEDS],
        "e4_shuffled_ctrl_acc": [all_results["runs"][f"seed_{s}"]["e4"]["four_classes"]["shuffled_label_control"]["accuracy"] for s in SEEDS],
        "e4_randfeat_ctrl_acc": [all_results["runs"][f"seed_{s}"]["e4"]["four_classes"]["random_feature_control"]["accuracy"] for s in SEEDS],
    }
    all_results["summary_stats"] = summary

    final_results_file = EXP_DIR / "results.json"
    with open(final_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 聚合多 seed 测量结果已写入 {final_results_file}")


if __name__ == "__main__":
    main()
