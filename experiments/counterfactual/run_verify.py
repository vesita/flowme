#!/usr/bin/env python3
"""阶段二（**独立进程 B**）：重新加载引擎与权重，真跑复核阶段一的每一条记录。

    uv run python experiments/counterfactual/run_verify.py

- C0：反事实 `X′` 复跑后是否仍等于阶段一记录的 `Y′`（= 编的反事实比率，PREREG §6）。
- C3：同一 (X, X′) 两阶段两次跑是否一致（必须 0 不一致）。
产出 `results/verify.json`。
"""
from __future__ import annotations

import json
import time

from common import PREREG, RESULTS, run_once, ykey


def main() -> int:
    prereg_mtime = PREREG.stat().st_mtime
    t0 = time.time()
    assert prereg_mtime < t0, "PREREG 必须早于首次运行"
    src = RESULTS / "search.json"
    search = json.loads(src.read_text(encoding="utf-8"))
    print(f"[读入] {src}  n_samples={search['n_samples']} "
          f"search 条目={len(search['search'])}", flush=True)

    from common import build_engine
    eng = build_engine()
    print(f"[引擎] device={eng.device} cards={sorted(eng.attached)} "
          f"(阶段一 cards={search['cards']})", flush=True)
    assert str(eng.device) == search["device"], "设备口径必须与阶段一一致"
    assert sorted(eng.attached) == search["cards"], "卡集合必须与阶段一一致"

    out: dict = {"verify_start": t0, "device": str(eng.device),
                 "src_mtime": src.stat().st_mtime, "baseline": [], "claims": []}

    # --- 基线复核（C3 之一） ---
    n_bad = 0
    for b in search["baseline"]:
        o = run_once(b["x"])
        ok = tuple(o["y"]) == tuple(b["y"])
        n_bad += 0 if ok else 1
        if not ok:
            out["baseline"].append({"idx": b["idx"], "x": b["x"],
                                    "y1": b["y"], "y2": o["y"]})
    out["baseline_checked"] = len(search["baseline"])
    out["baseline_mismatch"] = n_bad
    print(f"[基线] {len(search['baseline'])} 条复跑，不一致 {n_bad}", flush=True)

    # --- 反事实复核（C0 + C3） ---
    n_s = f_s = n_r1 = f_r1 = n_r2 = f_r2 = 0
    for rec in search["search"]:
        for kind in ("struct", "R1", "R2"):
            f = rec.get(kind)
            if not f or not f.get("found"):
                continue
            c = f["found"]
            o = run_once(c["x_prime"])
            ok = tuple(o["y"]) == tuple(c["y_prime"])
            n_s += kind == "struct"
            f_s += (kind == "struct" and not ok)
            n_r1 += kind == "R1"
            f_r1 += (kind == "R1" and not ok)
            n_r2 += kind == "R2"
            f_r2 += (kind == "R2" and not ok)
            if not ok:
                out["claims"].append({"idx": c["idx"], "family": c["family"],
                                      "kind": kind, "x": c["x"],
                                      "x_prime": c["x_prime"],
                                      "y_prime_stage1": c["y_prime"],
                                      "y_prime_stage2": list(o["y"])})
            # 同一 (X, X′) 里的 X 也复跑一遍，直接并入 C3
            ob = run_once(c["x"])
            if tuple(ob["y"]) != tuple(c["y"]):
                out["claims"].append({"idx": c["idx"], "family": c["family"],
                                      "kind": kind + "_X", "x": c["x"],
                                      "x_prime": c["x_prime"],
                                      "y_prime_stage1": c["y"],
                                      "y_prime_stage2": list(ob["y"])})

    out["struct_checked"], out["struct_fail"] = n_s, f_s
    out["R1_checked"], out["R1_fail"] = n_r1, f_r1
    out["R2_checked"], out["R2_fail"] = n_r2, f_r2
    out["verify_end"] = time.time()
    out["wall_sec"] = round(out["verify_end"] - t0, 2)
    path = RESULTS / "verify.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[C0] 结构化 {f_s}/{n_s} 失败；R1 {f_r1}/{n_r1}；R2 {f_r2}/{n_r2}；"
          f"基线不一致 {n_bad}", flush=True)
    print(f"[写出] {path}  用时 {out['wall_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
