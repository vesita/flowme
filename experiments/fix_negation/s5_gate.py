#!/usr/bin/env python3
"""S5 —— F5 老卡门禁：四张默认卡（combo_full、线上核）改前/改后 `evaluate_task` 逐样本口径。

用法：
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s5_gate.py --phase before
    （改 DEFAULT_ATTACH 之后）
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s5_gate.py --phase after
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s5_gate.py --compare

口径 = `prod_card_audit/a0_bases.py` 的 `combo_full/E_online`：`MultiTaskEngine(base_path=线上核,
auto_attach=True)` + `attach(DEFAULT_ATTACH)`，四老卡逐样本 `eval_persample`（与线上 `evaluate_task`
同公式）；门槛 = Δ(exact) ≥ −各自噪声带（PREREG F5）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.dont_write_bytecode = True

from common import (  # noqa: E402
    HERE, OLD_BANDS, ROOT, SEEDS, build_engine, eval_persample, make_loader, split_of,
)

from dtseek.tasks.dialogue import DEFAULT_ATTACH  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE  # noqa: E402

FOUR = ["pronoun", "relation", "sentiment", "person"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("before", "after"))
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args(argv)
    res_dir = HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        pb = json.loads((res_dir / "s5_before.json").read_text(encoding="utf-8"))
        pa = json.loads((res_dir / "s5_after.json").read_text(encoding="utf-8"))
        out: dict = {"before_attach": pb["attach"], "after_attach": pa["attach"], "cells": {}}
        ok = True
        for cap in FOUR:
            for S in SEEDS:
                b = pb["cells"][f"{cap}/s{S}"]
                a = pa["cells"][f"{cap}/s{S}"]
                d = {k: round(a[k] - b[k], 6) for k in ("exact_match", "cls_acc", "span_hit", "bg_fp")}
                gate = d["exact_match"] >= -OLD_BANDS[cap] - 1e-12
                ok = ok and gate
                out["cells"][f"{cap}/s{S}"] = {"before": b, "after": a, "delta": d,
                                               "band": OLD_BANDS[cap], "pass": gate}
                print(f"  {cap:10s} s{S} exact {b['exact_match']:.6f} → {a['exact_match']:.6f} "
                      f"Δ={d['exact_match']:+.6f} (带 −{OLD_BANDS[cap]:.4f}) "
                      f"cls Δ={d['cls_acc']:+.6f} {'PASS' if gate else 'FAIL'}")
        out["gate_all_pass"] = bool(ok)

        # 交叉核对：before 是否与 prod_card_audit a0（combo_full/E_online）逐位相同
        a0_path = ROOT / "experiments/prod_card_audit/results/a0_bases.json"
        if a0_path.exists():
            a0 = json.loads(a0_path.read_text(encoding="utf-8"))
            eq = {}
            for cap in FOUR:
                for S in SEEDS:
                    ref = a0["runs"]["combo_full"]["E_online"][cap][str(S)]
                    mine = pb["cells"][f"{cap}/s{S}"]
                    eq[f"{cap}/s{S}"] = (round(ref["exact_match"], 6) == round(mine["exact_match"], 6)
                                         and round(ref["cls_acc"], 6) == round(mine["cls_acc"], 6))
            out["cross_check_vs_prod_card_audit_a0"] = eq
            print(f"  [交叉核对] before 与 P15 a0（combo_full/E_online）逐位相同 = "
                  f"{all(eq.values())}（{sum(eq.values())}/{len(eq)}）")
        (res_dir / "s5_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        print(f"[F5] 四老卡门禁 {'全过' if ok else '有 FAIL'}；"
              f"[save] {res_dir / 's5_compare.json'}\nS5_COMPARE_DONE")
        return 0 if ok else 1

    t0 = time.time()
    attach = list(DEFAULT_ATTACH)
    eng = build_engine(attach[0], base_path=DEFAULT_BASE, auto_attach=True)
    from dtseek.tasks.plugin import resolve_tasks
    out: dict = {"prereg": "experiments/fix_negation/PREREG.md", "phase": args.phase,
                 "attach": attach, "base": DEFAULT_BASE, "combos": "combo_full",
                 "seeds": list(SEEDS), "cells": {}}
    print(f"[phase] {args.phase} attach={attach} base={DEFAULT_BASE}", flush=True)
    for cap in FOUR:
        spec = resolve_tasks([cap])[cap].spec
        dec, enc = eng.decoders[cap], eng.doc_encoder
        for S in SEEDS:
            ev, _ = split_of(cap, S)
            r = eval_persample(enc, dec, make_loader(ev, spec, bs=64), spec, eng.device)
            out["cells"][f"{cap}/s{S}"] = {
                "n": r["n"], "n_cls": r["n_cls"],
                "cls_acc": round(r["cls_acc"], 6), "exact_match": round(r["exact_match"], 6),
                "span_hit": round(r["span_hit"], 6), "bg_fp": round(r["bg_fp"], 6)}
            print(f"  [{cap} s{S}] exact={r['exact_match']:.6f} cls={r['cls_acc']:.6f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    del eng
    res = res_dir / f"s5_{args.phase}.json"
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res} time={time.time() - t0:.0f}s\nS5_{args.phase.upper()}_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
