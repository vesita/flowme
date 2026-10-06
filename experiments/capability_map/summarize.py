"""汇总：主表（5 能力 × 三档 cls_acc × 2 seed）、达成率、Spearman P1 判定、OOD 表。

输入：probe.json（探针）、eval.json（冻结/旁路/联合三档，含对齐自检）。
输出：summary.json + stdout 的 Markdown 表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

CAPS = ["pronoun", "sentiment", "relation", "person", "negation"]
SEEDS = (42, 43)
DENOM_MIN = 0.05          # PREREG §5：分母 < 0.05 或 ≤ 0 ⇒ 排除
RHO_MIN = 0.8
OOD_DROP_RED = 20.0       # PREREG §5：OOD 掉 > 20pt ⇒ 语体捷径
OOD_DROP_OK = 10.0


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    probe = load(HERE / "probe.json")
    ev = load(HERE / "eval.json")
    if not ev.get("align_all_ok"):
        print("!! 对齐自检失败（PREREG §3）—— 结论作废")
    out: dict = {"align_all_ok": ev.get("align_all_ok"), "caps": {}}

    print("\n## 主表（cls_acc 口径，逐 seed）\n")
    print("| 能力 | seed | 探针 cls_acc | 冻结 cls_acc | 旁路 cls_acc | 联合 cls_acc | 达成率 | 分母 | 备注 |")
    print("|---|---|---|---|---|---|---|---|---|")

    rows: dict[str, dict[int, dict]] = {c: {} for c in CAPS}
    for cap in CAPS:
        for S in SEEDS:
            p = probe["caps"][cap][str(S)]
            e = ev["caps"][cap][str(S)]
            pr = p["probe_cls_acc"]
            fr = e.get("frozen", {}).get("cls_acc")
            by = e.get("bypass", {}).get("cls_acc")
            jo = e.get("joint", {}).get("cls_acc")
            denom = (jo - fr) if (jo is not None and fr is not None) else None
            gain = ((by - fr) / denom) if (denom not in (None, 0) and by is not None) else None
            excl = denom is None or denom < DENOM_MIN
            note = "排除：分母<0.05" if excl and denom is not None and denom >= 0 else (
                   "排除：分母≤0" if excl else "")
            miss = [k for k in ("frozen", "bypass", "joint") if "missing" in e.get(k, {})]
            if miss:
                note = (note + " 缺:" + ",".join(miss)).strip()
            rows[cap][S] = {"probe": pr, "frozen": fr, "bypass": by, "joint": jo,
                            "denom": denom, "gain": gain, "excluded": excl, "note": note,
                            "probe_exact": p["probe_exact"],
                            "frozen_exact": e.get("frozen", {}).get("exact_match"),
                            "bypass_exact": e.get("bypass", {}).get("exact_match"),
                            "joint_exact": e.get("joint", {}).get("exact_match")}
            print(f"| {cap} | {S} | {_f(pr)} | {_f(fr)} | {_f(by)} | {_f(jo)} | "
                  f"{_f(gain)} | {_f(denom)} | {note} |")

    # ---- Spearman ----
    from scipy.stats import spearmanr
    rho: dict[int, dict] = {}
    for S in SEEDS:
        xs, ys, kept = [], [], []
        for cap in CAPS:
            r = rows[cap][S]
            if r["excluded"] or r["probe"] is None or r["gain"] is None:
                continue
            xs.append(r["probe"]); ys.append(r["gain"]); kept.append(cap)
        n = len(xs)
        if n >= 2:
            val = float(spearmanr(xs, ys).statistic)
        else:
            val = None
        rho[S] = {"rho": val, "n_valid": n, "kept": kept, "probe": xs, "gain": ys}
        print(f"\n### Spearman seed {S}: n_valid={n} kept={kept} rho={val}")

    # ---- P1 判定 ----
    ns = [rho[S]["n_valid"] for S in SEEDS]
    if min(ns) < 3:
        verdict = "P1 不可评估（n_valid ≤ 2，PREREG §5 可评估性下限）"
    else:
        vals = [rho[S]["rho"] for S in SEEDS]
        ok = [v is not None and abs(v) >= RHO_MIN and v > 0 for v in vals]
        verdict = "P1 过 ⇒ 探针可当加卡前筛查" if all(ok) else \
                  ("P1 单 seed 过 ⇒ 只报方向" if any(ok) else
                   "P1 不过 ⇒ 没有便宜筛查")
        if len(set(ok)) > 1:
            verdict += "（两 seed 不同号）"
    print(f"\n**P1 判定：{verdict}**")

    # ---- 口径 B（整句 exact）----
    print("\n## 口径 B：整句 exact（与口径 A 不同，不许混）\n")
    print("| 能力 | seed | 探针 exact | 冻结 exact | 旁路 exact | 联合 exact |")
    print("|---|---|---|---|---|---|")
    for cap in CAPS:
        for S in SEEDS:
            r = rows[cap][S]
            print(f"| {cap} | {S} | {_f(r['probe_exact'])} | {_f(r['frozen_exact'])} | "
                  f"{_f(r['bypass_exact'])} | {_f(r['joint_exact'])} |")

    # ---- OOD ----
    print("\n## P3：探针 OOD\n")
    print("| 能力 | seed | 分布内 cls_acc | probe_units OOD | Δpt | 判读 |")
    print("|---|---|---|---|---|---|")
    ood_rows = {}
    for cap in CAPS:
        for S in SEEDS:
            o = probe["caps"][cap][str(S)].get("ood_probe_units")
            if not o:
                continue
            d = o.get("drop_pt")
            verdict = ("语体捷径" if d is not None and d > OOD_DROP_RED else
                       "OOD 成立" if d is not None and d <= OOD_DROP_OK else "证据不足")
            ood_rows[f"{cap}/{S}"] = dict(o, verdict=verdict)
            print(f"| {cap} | {S} | {_f(o['in_dist_acc_real'])} | {_f(o['acc'])} | "
                  f"{_f(d)} | {verdict} |")

    print("\n### compose_ops 60 条词典外否定表达（委托指定 OOD）\n")
    print("| 能力 | seed | 集 | 分布内 | OOD | Δpt | 判读 |")
    print("|---|---|---|---|---|---|---|")
    compose = {}
    for cap in ("negation", "sentiment"):
        for S in SEEDS:
            o = probe["caps"][cap][str(S)].get("ood_compose60")
            if not o:
                continue
            for key, lab in (("cls_acc", "cls"), ("exact", "exact")):
                ind = o.get("in_dist_cls_acc" if key == "cls_acc" else "in_dist_exact")
                cur = o.get(key)
                d = o.get("drop_cls_pt" if key == "cls_acc" else "drop_exact_pt")
                verdict = ("语体捷径" if d is not None and d > OOD_DROP_RED else
                           "OOD 成立" if d is not None and d <= OOD_DROP_OK else "证据不足")
                compose[f"{cap}/{S}/{lab}"] = {"in": ind, "ood": cur, "drop_pt": d,
                                               "verdict": verdict, "n": o["n"]}
                print(f"| {cap} | {S} | {lab} | {_f(ind)} | {_f(cur)} | {_f(d)} | {verdict} |")

    out.update({"rows": rows, "spearman": {str(k): v for k, v in rho.items()},
                "p1_verdict": verdict, "ood_probe_units": ood_rows,
                "ood_compose60": compose})
    (HERE / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\n[save] {HERE / 'summary.json'}")
    return 0


def _f(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
