#!/usr/bin/env python3
"""S3 —— F3 默认行为等价性：改前/改后同批输入重放 + F1 前后对比（`respond` 口径）。

用法：
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s3_equiv.py --phase before
    （改 dialogue.py 的 DEFAULT_ATTACH 之后）
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s3_equiv.py --phase after
    PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s3_equiv.py --compare

投影定义见 PREREG.md §3；`--compare` 报两投影的不一致数（必须 = 0）。
negation 三项指标（M_detect/M_cls/M_span，含向量）也一并存，供 F1 前后配对 Δ 用。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.dont_write_bytecode = True

from common import (  # noqa: E402
    CAPS, FALLBACK_CARD, HERE, NEG, ROOT, SEEDS, build_engine, neg_rate_summary,
    neg_vectors, paired_delta, project, rate, run_respond, split_of_e2e,
)
from dtseek.tasks.dialogue import DEFAULT_ATTACH  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("before", "after"))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    res_dir = HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        """F3 分层（PREREG §3 + 本段注释）：
        L0 严格全记录比对（披露项，预期 ≠ 0：差异可能全部来自 negation 卡及其下游 conf/plan）；
        L1 **negation 证据相同**的记录 ⇒ **整条记录必须逐字相同**（不一致 = 0，硬门槛）；
        L2 negation 证据不同**的记录 ⇒ 非派生字段（type/cards_run/非 negation 证据）必须相同（= 0，硬门槛），
           text/plan/reason/kind/terminal 的差异只作归因计数。
        """
        pb, pa = (json.loads((res_dir / f"s3_{p}.json").read_text(encoding="utf-8"))
                  for p in ("before", "after"))
        out: dict = {"before_card": pb["attach"], "after_card": pa["attach"]}

        def neg_ev(r):
            return [e for e in r.get("evidence") or [] if e["card"] == NEG]

        def core(r):
            return {"type": r.get("type"), "cards_run": r.get("cards_run"),
                    "evidence": [e for e in r.get("evidence") or [] if e["card"] != NEG]}

        n = l0 = l1 = l2 = 0
        same_neg = 0
        diffs, attr = [], {"kind": 0, "terminal": 0, "reason": 0, "plan": 0, "text": 0}
        for key in pb["cells"]:
            rb, ra = pb["cells"][key]["records"], pa["cells"][key]["records"]
            assert len(rb) == len(ra), (key, len(rb), len(ra))
            for i, (x, y) in enumerate(zip(rb, ra)):
                n += 1
                if x != y:
                    l0 += 1
                if neg_ev(x) == neg_ev(y):
                    same_neg += 1
                    if x != y:
                        l1 += 1
                        if len(diffs) < 3:
                            diffs.append({"cell": key, "i": i, "before": x, "after": y})
                else:
                    if core(x) != core(y):
                        l2 += 1
                        if len(diffs) < 3:
                            diffs.append({"cell": key, "i": i, "before": x, "after": y})
                    for f in attr:
                        if x.get(f) != y.get(f):
                            attr[f] += 1
        out.update({"n_records": n, "l0_strict_full_record_mismatch": l0,
                    "l1_same_neg_evidence_records": same_neg,
                    "l1_mismatch": l1, "l2_diff_neg_evidence_records": n - same_neg,
                    "l2_core_mismatch": l2, "l2_attribute_counts": attr,
                    "examples": diffs})
        # F1 前后配对 Δ（negation 三项）
        out["paired_delta"] = {}
        for m in ("detect", "cls", "span"):
            out["paired_delta"][m] = {}
            for S in SEEDS:
                a = pb["neg"][str(S)]["vectors"][m]
                b = pa["neg"][str(S)]["vectors"][m]
                out["paired_delta"][m][str(S)] = paired_delta(a, b)  # after − before
        print(f"[F3] 记录 {n} 条")
        print(f"     L0 严格全记录不一致（披露）      = {l0}")
        print(f"     L1 negation 证据相同的记录 {same_neg} 条 ⇒ 全记录不一致 = **{l1}**（硬门槛）")
        print(f"     L2 negation 证据不同的记录 {n - same_neg} 条 ⇒ 非派生字段不一致 = **{l2}**（硬门槛）")
        print(f"     L2 归因计数（text/plan/reason/kind/terminal 差异条数）= {attr}")
        print("[F1] Δ = 改后 − 改前（配对，同批输入）")
        for m, d in out["paired_delta"].items():
            print(f"  M_{m:6s} " + "  ".join(
                f"s{S}: {d[str(S)]['delta']:+.4f}±{d[str(S)]['se']:.4f}"
                f"{'*' if d[str(S)]['significant'] else ''}" for S in SEEDS))
        print(f"  [before] {pb['attach']} / [after] {pa['attach']}")
        (res_dir / "s3_compare.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {res_dir / 's3_compare.json'}\nS3_COMPARE_DONE")
        return 0

    t0 = time.time()
    attach = list(DEFAULT_ATTACH)
    eng = build_engine(attach[0], base_path=DEFAULT_BASE)
    class_map = {c.name: i for i, c in enumerate(eng.specs[NEG].classes)}
    out: dict = {"prereg": "experiments/fix_negation/PREREG.md", "phase": args.phase,
                 "attach": attach, "base": DEFAULT_BASE, "seeds": list(SEEDS),
                 "caps": list(CAPS), "entry": "dtseek.tasks.dialogue.respond(..., type_='plain')",
                 "cells": {}, "neg": {}}
    print(f"[phase] {args.phase} attach={attach} base={DEFAULT_BASE}", flush=True)
    for cap in CAPS:
        for S in SEEDS:
            items = split_of_e2e(cap, S)
            if args.limit:
                items = items[: args.limit]
            recs = run_respond(eng, [it["text"] for it in items])
            out["cells"][f"{cap}/s{S}"] = {"n": len(items), "records": recs}
            if cap == NEG:
                v = neg_vectors(items, recs, class_map)
                out["neg"][str(S)] = {"n": len(items), "summary": neg_rate_summary(v),
                                      "vectors": v, "confusion": v["confusion"]}
                s = out["neg"][str(S)]["summary"]
                print(f"  [neg s{S}] n={len(items)} M_detect={s['detect']['rate']:.4f} "
                      f"M_cls={s['cls']['rate']:.4f} M_span={s['span']['rate']:.4f}", flush=True)
            else:
                print(f"  [{cap} s{S}] n={len(items)} ({time.time() - t0:.0f}s)", flush=True)
            del recs
    del eng
    res = res_dir / f"s3_{args.phase}.json"
    res.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {res} ({res.stat().st_size / 1e6:.1f} MB) time={time.time() - t0:.0f}s\n"
          f"S3_{args.phase.upper()}_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
