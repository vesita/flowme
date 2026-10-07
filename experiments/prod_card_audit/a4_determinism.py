#!/usr/bin/env python3
"""A4 —— 确定性：同一输入连跑两遍，逐样本比对，不一致条数必须 = 0。

用法：
    CUDA_VISIBLE_DEVICES="" uv run python experiments/prod_card_audit/a4_determinism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dtseek.tasks.plugin import resolve_tasks  # noqa: E402

from a0_bases import COMBOS, build_engine  # noqa: E402
from common import SEEDS, eval_persample, make_loader, split_of  # noqa: E402

N_HEAD = 64


def main() -> int:
    torch.set_num_threads(torch.get_num_threads())
    caps = sorted(set(sum(COMBOS.values(), [])))
    specs = {c: resolve_tasks([c])[c].spec for c in caps}
    out: dict = {"prereg": "experiments/prod_card_audit/PREREG.md",
                 "n_head": N_HEAD, "cells": []}
    total_mismatch = 0

    for combo, combo_caps in COMBOS.items():
        for base_key in ("E_own", "E_online"):
            eng = build_engine(base_key, combo)
            for cap in combo_caps:
                spec = specs[cap]
                for S in SEEDS:
                    ev, _ = split_of(cap, S)
                    loader = make_loader(ev[:N_HEAD], spec, bs=64)
                    a = eval_persample(eng.doc_encoder, eng.decoders[cap],
                                       loader, spec, eng.device)
                    b = eval_persample(eng.doc_encoder, eng.decoders[cap],
                                       loader, spec, eng.device)
                    mm_cls = sum(1 for x, y in zip(a["cls_ok"], b["cls_ok"]) if x != y)
                    mm_real = sum(1 for x, y in zip(a["cls_real"], b["cls_real"]) if x != y)
                    mm_ex = sum(1 for x, y in zip(a["exact_ok"], b["exact_ok"]) if x != y)
                    same_metric = (a["cls_acc"] == b["cls_acc"]
                                   and a["exact_match"] == b["exact_match"])
                    total_mismatch += mm_cls + mm_real + mm_ex
                    row = {"combo": combo, "base": base_key, "cap": cap, "seed": S,
                           "n": a["n"], "mismatch_cls": mm_cls,
                           "mismatch_real": mm_real, "mismatch_exact": mm_ex,
                           "metrics_identical": bool(same_metric),
                           "cls_acc": a["cls_acc"], "exact_match": a["exact_match"]}
                    out["cells"].append(row)
                    print(f"[{combo} {base_key} {cap} s{S}] n={a['n']} "
                          f"cls={a['cls_acc']:.4f} exact={a['exact_match']:.4f} "
                          f"mm(cls/real/exact)={mm_cls}/{mm_real}/{mm_ex} "
                          f"metrics_identical={same_metric}", flush=True)
            del eng

    out["total_mismatch"] = total_mismatch
    out["pass"] = bool(total_mismatch == 0)
    res = HERE / "results" / "a4_determinism.json"
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[A4] total_mismatch = {total_mismatch}  pass = {out['pass']}")
    print(f"[save] {res}")
    print("A4_DONE")
    return 0 if out["pass"] else 5


if __name__ == "__main__":
    sys.exit(main())
