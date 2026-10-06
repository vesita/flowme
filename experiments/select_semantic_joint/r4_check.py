#!/usr/bin/env python3
"""select_semantic_joint —— R4 基线：「不接本任务」时四张老卡的逐样本 pred/exact sha256。

配置 B（基线）= 核 ← `checkpoints/base_encoder.pt` + 四张老卡头 ← `capability_map/cards/*_frozen_s{S}.pt`，
**没有任何本任务部件**（无打分头、无 mean-pool、无 select 数据）。

主臂训练脚本在 step-0 会算配置 A 的同一组 sha256 并与本文件比对（R4 门禁）；
`analyze.py` 再对账一次。同时用 `evaluate_task` 与 `capability_map/summary.json` 双重对账。

用法：uv run python experiments/select_semantic_joint/r4_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    OLD_CARDS, RESULTS, SEEDS, old_card_baseline)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    out = {}
    for seed in SEEDS:
        rep = old_card_baseline(seed, "cuda" if _cuda() else "cpu")
        out[str(seed)] = rep
        print(f"R4_BASELINE s{seed} combined={rep['combined_sha256']}", flush=True)
        for n in OLD_CARDS:
            r = rep[n]
            print(f"  {n:9s} exact={r['exact']:.6f} eval_task={r['eval_task_exact']:.6f} "
                  f"capmap={r['capmap_frozen_exact']:.6f} Δ={r['delta_vs_capmap']:+.2e} "
                  f"sha256={r['sha256']}", flush=True)
            assert abs(r["delta_vs_capmap"]) < 1e-9, f"{n} 与 capability_map 基线对不上"
    (RESULTS / "r4_baseline_all.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for seed in SEEDS:
        (RESULTS / f"r4_baseline_s{seed}.json").write_text(
            json.dumps(out[str(seed)], ensure_ascii=False, indent=2), encoding="utf-8")
    print("[save] results/r4_baseline_s{42,43}.json + r4_baseline_all.json")
    print("R4_CHECK_DONE")
    return 0


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    sys.exit(main())
