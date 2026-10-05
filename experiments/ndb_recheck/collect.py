#!/usr/bin/env python
"""从 4 个训练进程的日志里抽 `AB_METRICS`，出原始表 + 配对 Δ + 代价表。

    uv run python experiments/ndb_recheck/collect.py --run-dir /tmp/ndb_recheck/run1

只读日志，不碰任何训练产物；输出 experiments/ndb_recheck/raw_metrics.json。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KEYS = ("repeat_mention_acc", "first_mention_acc", "cluster_f1", "exact_match",
        "id_acc", "cls_acc", "span_hit", "bg_fp", "n_measurable")
ARMS = ("base", "ndb")
SEEDS = (42, 43)


def load(run_dir: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            log = run_dir / f"person_{arm}_seed{seed}.log"
            rec = None
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.startswith("AB_METRICS "):
                    rec = json.loads(line[len("AB_METRICS "):])
            if rec is None:
                raise SystemExit(f"{log} 没有 AB_METRICS 行")
            out[(arm, seed)] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/tmp/ndb_recheck/run1")
    ap.add_argument("--json-out", default="experiments/ndb_recheck/raw_metrics.json")
    args = ap.parse_args()

    runs = load(Path(args.run_dir))
    print("================ 原始指标（逐 seed × 臂）================")
    for (arm, seed), rec in runs.items():
        m = rec["metrics"]
        cells = "  ".join(f"{k}={m.get(k, float('nan')):.4f}" for k in KEYS)
        print(f"{arm:>4} seed={seed}  {cells}")
        print(f"      训练墙钟 {rec['train_sec']:.1f}s | {rec['n_steps']} 步 | "
              f"{rec['sec_per_step']*1000:.1f} ms/步 | 峰值显存 {rec['peak_mem_mb']:.0f}MB | "
              f"头部参数 {rec['n_head_params']:,} | NDB参数 {rec['n_ndb_params']:,} | "
              f"表 {rec['ndb_table_gb']:.3f}GB")
        st = rec.get("ndb_stats") or {}
        if st:
            print(f"      NDB统计 last_gate={st.get('last_gate'):.4f} "
                  f"write_gate_bias={st.get('write_gate_bias'):.4f} "
                  f"read_gate_bias={st.get('read_gate_bias'):.4f} "
                  f"retrieval_top1_hit={st.get('retrieval_top1_hit'):.4f} "
                  f"n_written={st.get('n_written')}")

    print("\n================ 配对 Δ（ndb − base，同 seed）================")
    deltas: dict[str, list[float]] = {}
    for k in KEYS:
        ds = [runs[("ndb", s)]["metrics"][k] - runs[("base", s)]["metrics"][k] for s in SEEDS]
        deltas[k] = ds
        print(f"{k:>20}: " + "  ".join(f"{d:+.4f}" for d in ds) +
              f"   mean={statistics.fmean(ds):+.4f}  range={max(ds)-min(ds):.4f}")

    print("\n================ 代价（逐 seed）================")
    cost = {}
    for arm in ARMS:
        recs = [runs[(arm, s)] for s in SEEDS]
        cost[arm] = {
            "train_sec": [r["train_sec"] for r in recs],
            "sec_per_step": [r["sec_per_step"] for r in recs],
            "peak_mem_mb": [r["peak_mem_mb"] for r in recs],
            "n_head_params": recs[0]["n_head_params"],
            "n_ndb_params": recs[0]["n_ndb_params"],
            "ndb_table_gb": max(r.get("ndb_table_gb", 0.0) for r in recs),
        }
        print(f"{arm:>4}: 墙钟 {cost[arm]['train_sec']}s | "
              + "  ".join(f"{s*1000:.1f}ms/步" for s in cost[arm]["sec_per_step"])
              + " | 峰值显存 " + str([round(x) for x in cost[arm]["peak_mem_mb"]]) + "MB"
              + f" | 头参数 {cost[arm]['n_head_params']:,} NDB参数 {cost[arm]['n_ndb_params']:,}")
    d_sps = [cost["ndb"]["sec_per_step"][i] - cost["base"]["sec_per_step"][i] for i in range(2)]
    d_mem = [cost["ndb"]["peak_mem_mb"][i] - cost["base"]["peak_mem_mb"][i] for i in range(2)]
    print(f"Δ ms/步 = {[round(x*1000, 1) for x in d_sps]}  mean={statistics.fmean(d_sps)*1000:.1f}ms "
          f"({statistics.fmean(d_sps)/statistics.fmean(cost['base']['sec_per_step'])*100:+.1f}%)")
    print(f"Δ 峰值显存 = {[round(x) for x in d_mem]}MB  mean={statistics.fmean(d_mem):.0f}MB")

    out = {
        "runs": {f"{arm}_seed{s}": rec for (arm, s), rec in runs.items()},
        "deltas": deltas,
        "cost": cost,
        "delta_sec_per_step": d_sps,
        "delta_peak_mem_mb": d_mem,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
