#!/usr/bin/env python3
"""阶段一（进程 A）：基线 + 五族最小扰动搜索 + 同规模随机孪生。

    uv run python experiments/counterfactual/run_search.py [--limit N]

产出 `results/search.json`。Y′ 一律来自**本进程真跑**，绝不推断（PREREG §5）。
"""
from __future__ import annotations

import argparse
import json
import time

from common import (LOGS, MAX_CAND, PREREG, RESULTS, SEED0, CARDS,  # noqa: F401
                    load_samples, run_once, ykey)
from perturb import FAMILIES, family_candidates, lev, random_perturb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 句（冒烟用）")
    args = ap.parse_args()

    prereg_mtime = PREREG.stat().st_mtime
    t0 = time.time()
    assert prereg_mtime < t0, "PREREG 必须早于首次运行"
    print(f"[prereg] mtime={time.strftime('%F %T', time.localtime(prereg_mtime))} "
          f"< run_start={time.strftime('%F %T', time.localtime(t0))}", flush=True)

    sentences = load_samples(per_file=60)
    if args.limit:
        sentences = sentences[:args.limit]
    print(f"[样本] {len(sentences)} 句（card_flow PREREG §1 规则，seed=42）", flush=True)

    from common import build_engine
    eng = build_engine()
    print(f"[引擎] device={eng.device} cards={sorted(eng.attached)}", flush=True)

    out: dict = {
        "prereg_mtime": prereg_mtime, "run_start": t0, "device": str(eng.device),
        "cards": sorted(eng.attached), "max_cand": MAX_CAND, "seed0": SEED0,
        "n_samples": len(sentences), "baseline": [], "search": [],
    }

    t_base = time.time()
    for i, x in enumerate(sentences):
        b = run_once(x)
        out["baseline"].append({"idx": i, "x": x, "y": b["y"], "ch": b["ch"],
                                "skeleton": b.get("skeleton"),
                                "n_valid": b.get("n_valid")})
    print(f"[基线] {len(sentences)} 句用时 {time.time()-t_base:.1f}s；"
          f"ok={sum(1 for b in out['baseline'] if b['ch']=='ok')}", flush=True)

    n_run = 0
    for i, x in enumerate(sentences):
        y0 = tuple(out["baseline"][i]["y"])
        for fam in FAMILIES:
            cands = family_candidates(x, fam)
            K = min(MAX_CAND, len(cands))
            schedule = [lev(x, c) for c in cands[:K]]
            rec = {"idx": i, "family": fam, "n_cands": len(cands), "budget_K": K,
                   "schedule": schedule, "exhausted": len(cands) <= K}
            if K == 0:
                rec["note"] = "no_candidates"
                out["search"].append(rec)
                continue

            # --- 结构化搜索：按大小递增，第一个改变即停 ---
            st_runs, found = [], None
            for j, xp in enumerate(cands[:K]):
                o = run_once(xp)
                n_run += 1
                ch = ykey(o) != y0
                st_runs.append([j, schedule[j], 1 if ch else 0]
                               + ([o["y"]] if ch else []))
                if ch:
                    found = {"idx": i, "family": fam, "kind": "struct", "attempt": j,
                             "lev": schedule[j], "x": x, "x_prime": xp,
                             "y": list(y0), "y_prime": o["y"],
                             "ch_before": out["baseline"][i]["ch"], "ch_after": o["ch"]}
                    break
            rec["struct"] = {"runs": st_runs, "found": found,
                             "n_run": len(st_runs)}

            # --- 随机孪生：同预算 K、同规模序列、同停止规则 ---
            for mode in ("R1", "R2"):
                r_runs, r_found, unreachable = [], None, 0
                for j, size in enumerate(schedule):
                    xp = random_perturb(x, size, SEED0, i, fam, mode, j)
                    if xp is None:
                        unreachable += 1
                        r_runs.append([j, size, 0])
                        continue
                    assert xp != x and lev(x, xp) == size, (mode, j, size)
                    o = run_once(xp)
                    n_run += 1
                    ch = ykey(o) != y0
                    r_runs.append([j, size, 1 if ch else 0] + ([o["y"]] if ch else []))
                    if ch:
                        r_found = {"idx": i, "family": fam, "kind": mode, "attempt": j,
                                   "lev": size, "x": x, "x_prime": xp,
                                   "y": list(y0), "y_prime": o["y"],
                                   "ch_before": out["baseline"][i]["ch"],
                                   "ch_after": o["ch"]}
                        break
                rec[mode] = {"runs": r_runs, "found": r_found, "n_run": len(r_runs),
                             "unreachable": unreachable}

            out["search"].append(rec)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sentences)}] 累计前向 {n_run} 次 "
                  f"用时 {time.time()-t_base:.0f}s", flush=True)

    out["run_end"] = time.time()
    out["n_forward"] = n_run
    path = RESULTS / "search.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[写出] {path}  前向 {n_run} 次，总用时 {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
