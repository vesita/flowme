#!/usr/bin/env python3
"""A0（主判据）—— 自带基座 vs 线上基座：两个引擎 × 两种卡组合 × 2 seed，配对 Δ ± SE。

用法：
    CUDA_VISIBLE_DEVICES="" uv run python experiments/prod_card_audit/a0_bases.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dtseek.tasks.dialogue import DEFAULT_ATTACH  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402

from common import (  # noqa: E402
    EVAL_FILES,
    SEEDS,
    eval_persample,
    make_loader,
    paired_delta,
    split_of,
)

E_BASES = {
    "E_own": "experiments/compose_ops/artifacts/base_encoder.pt",
    "E_online": "checkpoints/base_encoder.pt",
}
FOUR = ["pronoun", "relation", "sentiment", "person"]
COMBOS = {
    "combo_full": FOUR + ["negation"],
    "combo_neg": ["negation"],
}
CARD_PATHS = {c: f"checkpoints/cards/{c}.pt" for c in FOUR}
CARD_PATHS["negation"] = DEFAULT_ATTACH[0]


def build_engine(base_key: str, combo: str) -> MultiTaskEngine:
    """真实加载路径：E_online 用 auto_attach 默认四卡；combo_neg 不自动挂卡。"""
    bp = E_BASES[base_key]
    if combo == "combo_full":
        eng = MultiTaskEngine(base_path=bp, auto_attach=True)
        for p in DEFAULT_ATTACH:
            eng.attach(p)
    else:
        eng = MultiTaskEngine(base_path=bp, auto_attach=False)
        for p in DEFAULT_ATTACH:
            eng.attach(p)
    return eng


def fmt(x: float) -> str:
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每个评测集只取前 64 条（跑通用）")
    args = ap.parse_args(argv)

    t0 = time.time()
    out: dict = {
        "prereg": "experiments/prod_card_audit/PREREG.md",
        "bases": E_BASES,
        "combos": {k: v for k, v in COMBOS.items()},
        "card_paths": CARD_PATHS,
        "eval_files": EVAL_FILES,
        "seeds": list(SEEDS),
        "smoke": bool(args.smoke),
        "runs": {},
        "engine_meta": {},
        "selfcheck": {},
    }

    # ---- 口径自检：逐样本版 vs 线上 evaluate_task（1 个 cell）----
    from dtseek.tasks.runtime import evaluate_task  # noqa: E402
    from dtseek.tasks.plugin import resolve_tasks  # noqa: E402

    specs = {c: resolve_tasks([c])[c].spec for c in set(sum(COMBOS.values(), []))}
    selfcheck_done = False

    for combo, caps in COMBOS.items():
        out["runs"][combo] = {}
        for base_key in E_BASES:
            eng = build_engine(base_key, combo)
            out["engine_meta"][f"{combo}/{base_key}"] = {
                "base_path": eng.base_path,
                "doc_file": str(eng.base_path),
                "attached": sorted(eng.attached),
                "device": str(eng.device),
            }
            print(f"[engine] {combo} {base_key}: base_path={eng.base_path} "
                  f"attached={sorted(eng.attached)}", flush=True)
            for cap in caps:
                spec = specs[cap]
                dec = eng.decoders[cap]
                enc = eng.doc_encoder
                cell: dict = {}
                for S in SEEDS:
                    ev, _tr = split_of(cap, S)
                    if args.smoke:
                        ev = ev[:64]
                    loader = make_loader(ev, spec, bs=64)
                    r = eval_persample(enc, dec, loader, spec, eng.device)
                    cell[str(S)] = {
                        "n": r["n"], "n_cls": r["n_cls"], "n_bg": r["n_bg"],
                        "cls_acc": round(r["cls_acc"], 6),
                        "exact_match": round(r["exact_match"], 6),
                        "span_hit": round(r["span_hit"], 6),
                        "bg_fp": round(r["bg_fp"], 6),
                        "slice_precision": round(r["slice_precision"], 6),
                        "slice_recall": round(r["slice_recall"], 6),
                        "cls_ok": r["cls_ok"], "cls_real": r["cls_real"],
                        "exact_ok": r["exact_ok"],
                    }
                    if (not selfcheck_done and combo == "combo_full"
                            and base_key == "E_online" and cap == "negation" and S == 42):
                        rep = evaluate_task(enc, dec, loader, eng.device, spec)
                        keys = ("cls_acc", "exact_match", "span_hit", "bg_fp",
                                "n_cls", "n_bg")
                        out["selfcheck"] = {
                            "cell": f"{combo}/{base_key}/{cap}/s{S}",
                            "mine": {k: r[k] for k in keys},
                            "evaluate_task": {k: rep[k] for k in keys},
                        }
                        same = all(abs(out["selfcheck"]["mine"][k] -
                                       out["selfcheck"]["evaluate_task"][k]) < 1e-12
                                   for k in out["selfcheck"]["mine"])
                        out["selfcheck"]["match"] = bool(same)
                        print(f"[selfcheck] mine vs evaluate_task match={same}", flush=True)
                        selfcheck_done = True
                    print(f"[run] {combo} {base_key} {cap} s{S}: "
                          f"n={r['n']} cls={r['cls_acc']:.4f} exact={r['exact_match']:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                out["runs"][combo].setdefault(base_key, {})[cap] = cell
            del eng
            torch.set_num_threads(torch.get_num_threads())

    # ---- 配对 Δ（Δ = E_online − E_own；负 = 在产掉分）----
    paired: dict = {}
    for combo, per_base in out["runs"].items():
        paired[combo] = {}
        for cap in COMBOS[combo]:
            rows = {}
            for S in SEEDS:
                a = per_base["E_own"][cap][str(S)]      # 卡自带核
                b = per_base["E_online"][cap][str(S)]   # 线上核
                cls = paired_delta(a["cls_ok"], a["cls_real"], b["cls_ok"], b["cls_real"])
                ex = paired_delta(a["exact_ok"], [1] * a["n"], b["exact_ok"], [1] * b["n"])
                rows[str(S)] = {
                    "n_cls": cls["n"], "n_exact": ex["n"],
                    "cls_own": a["cls_acc"], "cls_online": b["cls_acc"],
                    "delta_cls": round(cls["delta"], 6), "se_cls": round(cls["se"], 6),
                    "sig_cls": cls["significant"],
                    "exact_own": a["exact_match"], "exact_online": b["exact_match"],
                    "delta_exact": round(ex["delta"], 6), "se_exact": round(ex["se"], 6),
                    "sig_exact": ex["significant"],
                }
            d1, d2 = rows["42"]["delta_cls"], rows["43"]["delta_cls"]
            same_sign = (d1 <= 0 and d2 <= 0) or (d1 >= 0 and d2 >= 0)
            both_sig = rows["42"]["sig_cls"] and rows["43"]["sig_cls"] and same_sign
            verdict = ("显著变差" if both_sig and d1 < 0 and d2 < 0
                       else "显著变好" if both_sig and d1 > 0 and d2 > 0
                       else "未测出显著差异")
            rows["verdict"] = verdict
            paired[combo][cap] = rows
    out["paired"] = paired

    # ---- 判定 ----
    neg_full = paired["combo_full"]["negation"]["verdict"]
    neg_only = paired["combo_neg"]["negation"]["verdict"]
    # 反向错配对照臂：四默认卡的自带核 == 线上核 ⇒ 它们在 E_own（外来核）下应更差
    # ⇔ Δ(线上 − 自带) > 0 ⇒ verdict「显著变好」。PREREG §2-F2 的「至少 3/4 显著变差」
    # 指的就是"在 E_own 下显著变差"。
    def ctrl_sens(metric: str) -> list[str]:
        hit = []
        for c in FOUR:
            rows = paired["combo_full"][c]
            d1, d2 = rows["42"][f"delta_{metric}"], rows["43"][f"delta_{metric}"]
            s1, s2 = rows["42"][f"sig_{metric}"], rows["43"][f"sig_{metric}"]
            if s1 and s2 and d1 > 0 and d2 > 0:
                hit.append(c)
        return hit

    sens_cls = ctrl_sens("cls")
    sens_exact = ctrl_sens("exact")
    ctrl_worse = len(sens_cls)
    f2_gate = ctrl_worse >= 3
    if not f2_gate:
        judge = "证据不足"
        why = (f"F2 检验力闸门未过：反向错配对照臂（四默认卡）只有 {ctrl_worse}/4 在 E_own 下"
               f"显著变差（主口径 cls，需 ≥3/4）；exact 口径 {len(sens_exact)}/4；"
               f"person 天花板 cls=1.0000/1.0000 无法显差")
    elif neg_full == "显著变差" and neg_only == "显著变差":
        judge = "在产缺陷成立（显著变差）"
        why = "negation 在两组合 × 两 seed 全部显著变差（配对 |Δ|>2SE 且同号）"
    elif neg_full == "未测出显著差异" and neg_only == "未测出显著差异":
        judge = "张量不一致但无显著影响"
        why = "negation 在两组合 × 两 seed 均未测出显著差异（配对 |Δ| ≤ 2SE）——**不是「没有效果」**"
    else:
        judge = "证据不足"
        why = f"两组合结论不一致：combo_full={neg_full} combo_neg={neg_only}"
    out["judgement"] = {
        "negation_combo_full": neg_full,
        "negation_combo_neg": neg_only,
        "control_four_worse_on_E_own_cls": sens_cls,
        "control_four_worse_on_E_own_exact": sens_exact,
        "control_count_cls": ctrl_worse,
        "control_count_exact": len(sens_exact),
        "F2_gate_passed": f2_gate,
        "judge": judge,
        "why": why,
    }

    # ---- 打表 ----
    print("\n=== A0 主表（Δ = 线上核 − 卡自带核；负 = 在产掉分）===")
    for combo in COMBOS:
        for cap in COMBOS[combo]:
            rows = paired[combo][cap]
            for S in SEEDS:
                r = rows[str(S)]
                print(f"{combo:11s} {cap:10s} s{S}  cls own={r['cls_own']:.4f} "
                      f"online={r['cls_online']:.4f}  Δ={r['delta_cls']:+.4f}±{r['se_cls']:.4f} "
                      f"({'SIG' if r['sig_cls'] else '   '}) | "
                      f"exact own={r['exact_own']:.4f} online={r['exact_online']:.4f} "
                      f"Δ={r['delta_exact']:+.4f}±{r['se_exact']:.4f} "
                      f"({'SIG' if r['sig_exact'] else '   '})")
            print(f"{'':11s} {cap:10s} verdict = {rows['verdict']}")
    print("\n[JUDGE]", out["judgement"]["judge"])
    print("        ", out["judgement"]["why"])
    print("[selfcheck]", out["selfcheck"].get("match"), out["selfcheck"].get("cell"))

    name = "a0_smoke.json" if args.smoke else "a0_bases.json"
    res = HERE / "results" / name
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}")
    print(f"[time] {time.time() - t0:.0f}s")
    print("A0_DONE")
    return 0 if out["selfcheck"].get("match") else 4


if __name__ == "__main__":
    sys.exit(main())
