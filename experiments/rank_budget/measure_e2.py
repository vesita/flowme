#!/usr/bin/env python3
"""E2: 每张卡的头部容量占用。

测法：
复用/扩展 scripts/analyze_card_rank.py 的分析逻辑，对 checkpoints/cards/*.pt 中各卡解码器权重
做 SVD 有效秩分析，汇总每张卡的 util 分布（min, mean, median, max, <30% 占比, >70% 占比）。

判据：util < 30% = 容量被浪费；util > 70% = 这张卡在吃力。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_card_rank import analyze  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards-dir", default="checkpoints/cards")
    parser.add_argument("--out", default="experiments/rank_budget/e2_results.json")
    args = parser.parse_args()

    cards_dir = Path(args.cards_dir)
    card_files = sorted(cards_dir.glob("*.pt"))
    if not card_files:
        raise FileNotFoundError(f"未找到任务卡文件在 {cards_dir}")

    results = {}
    print(f"[E2] 分析 {len(card_files)} 张任务卡: {[f.name for f in card_files]}")

    for f in card_files:
        r = analyze(str(f))
        task = r["task"]
        utils = [row["util"] for row in r["rows"]]
        n_low = sum(1 for u in utils if u < 0.30)
        n_high = sum(1 for u in utils if u > 0.70)
        n_mid = sum(1 for u in utils if 0.30 <= u <= 0.70)
        summary = {
            "task": task,
            "classes": r["classes"],
            "total_decoder_params": r["total"],
            "num_2d_matrices": len(utils),
            "util_min": min(utils),
            "util_max": max(utils),
            "util_mean": sum(utils) / len(utils),
            "n_below_30": n_low,
            "n_above_70": n_high,
            "n_mid_30_70": n_mid,
            "pct_above_70": n_high / len(utils),
            "rows": r["rows"],
        }
        results[task] = summary
        print(f"  任务 {task:10s} | 矩阵数: {len(utils):2d} | util 范围: {min(utils)*100:5.1f}% ~ {max(utils)*100:5.1f}% | 均值: {summary['util_mean']*100:5.1f}% | >70% 占比: {n_high}/{len(utils)} ({summary['pct_above_70']*100:4.1f}%) | <30%: {n_low}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2, ensure_ascii=False)
        print(f"[E2] 结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
