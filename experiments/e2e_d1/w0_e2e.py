#!/usr/bin/env python3
"""E0（主）+ W3 + W4 —— 端到端对话路径（`dialogue.respond`）在两种基座下的实测。

用法：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/e2e_d1/w0_e2e.py [--smoke] [--limit N]

口径（PREREG §2 写死）：
  * 真实入口 = `respond(engine, text, type_="plain")`（内部 `run_cards` → `engine.predict` 分段解码）；
  * 两引擎只换 base_path，卡文件完全相同；
  * 输入 = capability_map cache 的 eval_S（与 prod_card_audit 同公式），报文件清单。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter

from common import (  # noqa: E402
    CAPS, E_BASES, EVAL_FILES, HERE, ROOT, SEEDS,
    anchor_key, paired_delta, split_of, truth_first, verdict_two_seed,
)

import torch  # noqa: E402

from dtseek.tasks.dialogue import DEFAULT_ATTACH, respond  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402

NEG = "negation"
MOOD = "sentiment"
ENGINE_KEYS = tuple(E_BASES)


def build_engine(base_rel: str) -> MultiTaskEngine:
    """与 `examples/example_dialogue.py:54-56` 逐字同构：默认四卡 + DEFAULT_ATTACH。"""
    eng = MultiTaskEngine(base_path=str(ROOT / base_rel))
    for path in DEFAULT_ATTACH:
        eng.attach(path)
    return eng


def run_engine(eng, texts: list[str]) -> list[dict]:
    out = []
    for t in texts:
        try:
            out.append(respond(eng, t, type_="plain"))
        except Exception as exc:  # noqa: BLE001 —— respond 本应 fail-closed；真抛了要记下来
            out.append({"kind": "CRASH", "text": f"{type(exc).__name__}: {exc}",
                        "evidence": [], "cards_run": [], "plan": [], "terminal": "crash",
                        "reason": str(exc), "type": "plain"})
    return out


def binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / max(1, n))


def rate(vec: list[int]) -> float:
    return sum(vec) / max(1, len(vec))


def neg_metrics(items, recs, class_map) -> dict:
    """M2/M3/M4 + 配对 Δ（Δ = A 线上 − B 来源核；负 = 线上核更差）。"""
    truths = [truth_first(it) for it in items]
    cores = [it["text"].strip() for it in items]
    real = [t["label"] > 0 for t in truths]
    rec_has = {k: [any(e["card"] == NEG for e in recs[k][i]["evidence"])
                   for i in range(len(items))] for k in ENGINE_KEYS}

    vec: dict[str, dict[str, list[int]]] = {k: {"detect": [], "cls": [], "span": []}
                                            for k in ENGINE_KEYS}
    for k in ENGINE_KEYS:
        for i, rec in enumerate(recs[k]):
            ev = [e for e in rec["evidence"] if e["card"] == NEG]
            has = bool(ev)
            vec[k]["detect"].append(int(has == real[i]))
            if not real[i]:
                continue
            pred_id = class_map.get(ev[0]["class_name"], -1) if has else 0
            vec[k]["cls"].append(int(pred_id == truths[i]["label"]))
            if truths[i]["aligned"]:           # 两侧同掩码 ⇒ 配对长度一致
                if has:
                    s_, e_ = ev[0]["span"]
                    vec[k]["span"].append(int(cores[i][s_:e_ + 1] == truths[i]["sub"]))
                else:
                    vec[k]["span"].append(0)

    paired = {m: paired_delta(vec["B_own"][m], vec["A_online"][m])
              for m in ("detect", "cls", "span")}

    # 混淆 + 两引擎分歧方向（把 Δ 拆成「线上漏检」还是「线上误报」）
    def confusion(k: str) -> dict:
        tp = fn = fp = tn = 0
        for i, has in enumerate(rec_has[k]):
            if real[i] and has:
                tp += 1
            elif real[i]:
                fn += 1
            elif has:
                fp += 1
            else:
                tn += 1
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}

    disagree = {
        "real_B_has_A_not": sum(1 for i in range(len(items))
                                if real[i] and rec_has["B_own"][i] and not rec_has["A_online"][i]),
        "real_A_has_B_not": sum(1 for i in range(len(items))
                                if real[i] and rec_has["A_online"][i] and not rec_has["B_own"][i]),
        "bg_A_has_B_not": sum(1 for i in range(len(items))
                              if not real[i] and rec_has["A_online"][i] and not rec_has["B_own"][i]),
        "bg_B_has_A_not": sum(1 for i in range(len(items))
                              if not real[i] and rec_has["B_own"][i] and not rec_has["A_online"][i]),
    }

    return {
        "n": len(items), "n_real": sum(real), "n_bg": len(items) - sum(real),
        "n_span_scored": len(vec["A_online"]["span"]),
        "rates": {k: {m: rate(vec[k][m]) for m in ("detect", "cls", "span")}
                  for k in ENGINE_KEYS},
        "confusion": {k: confusion(k) for k in ENGINE_KEYS},
        "disagree": disagree,
        "paired": paired,
        "vectors": vec,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每个 cell 只取前 32 条（跑通用）")
    ap.add_argument("--limit", type=int, default=None, help="每 cell 条数上限")
    ap.add_argument("--caps", default=None, help="逗号分隔的子集（默认 5 个全跑）")
    args = ap.parse_args(argv)

    caps = tuple(args.caps.split(",")) if args.caps else CAPS
    limit = args.limit or (32 if args.smoke else None)
    tag = "w0_smoke" if (args.smoke or args.limit) else "w0_e2e"

    t0 = time.time()
    out: dict = {
        "prereg": "experiments/e2e_d1/PREREG.md",
        "entry": "dtseek.tasks.dialogue.respond(engine, text, type_='plain')",
        "inner_decoder": "engine.predict -> _run_segment（分段解码；与 evaluate_task 整段 loader 口径不同，不许混报）",
        "bases": E_BASES, "caps": list(caps), "seeds": list(SEEDS),
        "eval_files": sorted(f for c in caps for f in EVAL_FILES if f.startswith(c + "_")),
        "limit": limit, "cells": {}, "trigger": {}, "determinism": {},
    }

    engines = {k: build_engine(rel) for k, rel in E_BASES.items()}
    class_map = {c.name: i for i, c in enumerate(engines["A_online"].specs[NEG].classes)}
    out["negation_class_map"] = class_map
    print(f"[entry] {out['entry']}", flush=True)
    print(f"[class_map] {class_map}", flush=True)

    for cap in caps:
        for S in SEEDS:
            items = split_of(cap, S)
            if limit:
                items = items[:limit]
            texts = [it["text"] for it in items]
            n = len(items)
            recs = {k: run_engine(engines[k], texts) for k in ENGINE_KEYS}

            # ---- W3 影响面：negation 卡是否真被调用 + 出口分布 ----
            out["trigger"][f"{cap}/s{S}"] = {
                k: {"n": n,
                    "called": sum(1 for r in recs[k] if NEG in r["cards_run"]),
                    "ratio": sum(1 for r in recs[k] if NEG in r["cards_run"]) / max(1, n),
                    "kind": dict(Counter(r["kind"] for r in recs[k])),
                    "terminal": dict(Counter(r["terminal"] for r in recs[k])),
                    "crash": sum(1 for r in recs[k] if r["kind"] == "CRASH")}
                for k in ENGINE_KEYS
            }

            # ---- M1 逐条相同比例 + M5 逐卡证据一致性 ----
            def frac(pred) -> float:
                return sum(1 for i in range(n) if pred(i)) / max(1, n)

            same_rec = sum(1 for i in range(n) if recs["A_online"][i] == recs["B_own"][i])
            cell = {
                "n": n,
                "same_record": same_rec,
                "same_record_ratio": same_rec / max(1, n),
                "same_record_se": binom_se(same_rec / max(1, n), n),
                "same_text_ratio": frac(lambda i: recs["A_online"][i]["text"] ==
                                        recs["B_own"][i]["text"]),
                "same_evidence_ratio": frac(lambda i: recs["A_online"][i]["evidence"] ==
                                            recs["B_own"][i]["evidence"]),
                "same_cards_run_ratio": frac(lambda i: recs["A_online"][i]["cards_run"] ==
                                             recs["B_own"][i]["cards_run"]),
                "same_kind_ratio": frac(lambda i: recs["A_online"][i]["kind"] ==
                                        recs["B_own"][i]["kind"]),
                "neg_anchor_same_ratio": frac(
                    lambda i: anchor_key(recs["A_online"][i], NEG, texts[i])
                    == anchor_key(recs["B_own"][i], NEG, texts[i])),
                "mood_anchor_same_ratio": frac(
                    lambda i: anchor_key(recs["A_online"][i], MOOD, texts[i])
                    == anchor_key(recs["B_own"][i], MOOD, texts[i])),
            }
            if cap == NEG:
                cell["neg"] = neg_metrics(items, recs, class_map)
            out["cells"][f"{cap}/s{S}"] = cell
            print(f"[cell] {cap} s{S}: n={n} same={cell['same_record_ratio']:.4f} "
                  f"negAnchor={cell['neg_anchor_same_ratio']:.4f} "
                  f"moodAnchor={cell['mood_anchor_same_ratio']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

            # ---- W4 确定性：同输入连跑两遍，整条记录比对 ----
            k4 = min(64, n)
            for key in ENGINE_KEYS:
                again = run_engine(engines[key], texts[:k4])
                bad = sum(1 for x, y in zip(recs[key][:k4], again) if x != y)
                out["determinism"].setdefault(f"{cap}/s{S}", {})[key] = {"n": k4, "mismatch": bad}
            del recs

    # ---- 汇总：M2/M3/M4 两 seed verdict ----
    summary: dict = {}
    if NEG in caps:
        for metric in ("detect", "cls", "span"):
            d = {str(S): out["cells"][f"{NEG}/s{S}"]["neg"]["paired"][metric] for S in SEEDS}
            summary[metric] = {
                "per_seed": {k: {"n": v["n"], "delta": round(v["delta"], 6),
                                 "se": round(v["se"], 6), "sig": v["significant"]}
                             for k, v in d.items()},
                "verdict": verdict_two_seed(d[str(SEEDS[0])], d[str(SEEDS[1])]),
            }
    out["paired_neg"] = summary

    trig = {"called": 0, "n": 0}
    for v in out["trigger"].values():
        for k in ENGINE_KEYS:
            trig["called"] += v[k]["called"]
            trig["n"] += v[k]["n"]
    out["trigger_total"] = {**trig, "ratio": trig["called"] / max(1, trig["n"])}
    det = {"n": 0, "mismatch": 0}
    for c in out["determinism"].values():
        for v in c.values():
            det["n"] += v["n"]
            det["mismatch"] += v["mismatch"]
    out["determinism_total"] = det

    # ---- 打表 ----
    print("\n=== E0 主表（端到端 respond；same = 两引擎整条记录相同的比例）===")
    for key, cell in out["cells"].items():
        print(f"{key:18s} n={cell['n']:5d} same_record={cell['same_record_ratio']:.4f}"
              f"±{cell['same_record_se']:.4f}  same_text={cell['same_text_ratio']:.4f}"
              f"  same_ev={cell['same_evidence_ratio']:.4f}"
              f"  same_kind={cell['same_kind_ratio']:.4f}"
              f"  negAnchorSame={cell['neg_anchor_same_ratio']:.4f}"
              f"  moodAnchorSame={cell['mood_anchor_same_ratio']:.4f}")
        if "neg" in cell:
            g = cell["neg"]
            for m in ("detect", "cls", "span"):
                p = g["paired"][m]
                print(f"{'':18s}   M_{m:6s} A(线上)={g['rates']['A_online'][m]:.4f} "
                      f"B(来源核)={g['rates']['B_own'][m]:.4f}  "
                      f"Δ={p['delta']:+.4f}±{p['se']:.4f} n={p['n']} "
                      f"({'SIG' if p['significant'] else '   '})")
            for k in ENGINE_KEYS:
                c = g["confusion"][k]
                print(f"{'':18s}     conf {k:9s} tp={c['tp']} fn={c['fn']} "
                      f"fp={c['fp']} tn={c['tn']}")
            print(f"{'':18s}     分歧方向 {g['disagree']}")

    # ---- W3：出口分布（kind）按引擎汇总 ----
    agg_kind = {k: Counter() for k in ENGINE_KEYS}
    agg_term = {k: Counter() for k in ENGINE_KEYS}
    for v in out["trigger"].values():
        for k in ENGINE_KEYS:
            agg_kind[k].update(v[k]["kind"])
            agg_term[k].update(v[k]["terminal"])
    out["kind_total"] = {k: dict(agg_kind[k]) for k in ENGINE_KEYS}
    out["terminal_total"] = {k: dict(agg_term[k]) for k in ENGINE_KEYS}
    print("\n[W3] 出口分布（全能力全 seed 汇总）")
    for k in ENGINE_KEYS:
        print(f"  {k:9s} kind={out['kind_total'][k]}  terminal={out['terminal_total'][k]}")
    print("\n=== E1 verdict（两 seed 同号且 |Δ| > 2SE）===")
    for m, v in summary.items():
        print(f"  {m:7s} {v['verdict']}   "
              + "  ".join(f"s{k}: Δ={r['delta']:+.4f}±{r['se']:.4f}{'*' if r['sig'] else ''}"
                          for k, r in v["per_seed"].items()))
    print(f"\n[W3] negation 触发总计 {trig['called']}/{trig['n']} = {out['trigger_total']['ratio']:.4f}")
    print(f"[W4] 确定性 mismatch {det['mismatch']}/{det['n']}")
    print(f"[time] {time.time() - t0:.0f}s")

    res = HERE / "results" / f"{tag}.json"
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}")
    print("W0_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
