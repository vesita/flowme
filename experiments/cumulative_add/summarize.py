"""summarize.py —— 汇总 cumulative_add 的全部产物，判 P1/P2/P3（判据逐字来自 PREREG.md）。

只读 `cards/*.json`、`bands.json`、`experiments/core_keep/cards/J3_*_metrics.json`，
写 `results.json`，并打印主表（Markdown）。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (42, 43)
OLD = ["pronoun", "sentiment", "relation", "person"]
STEP_CARDS = {
    1: ["pronoun", "sentiment", "relation", "person", "negation"],
    2: ["pronoun", "sentiment", "relation", "person", "negation", "idiom"],
    3: ["pronoun", "sentiment", "relation", "person", "negation", "idiom", "ownership"],
}
NEW = {1: "negation", 2: "idiom", 3: "ownership"}
SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000, "person": 6000,
           "negation": 6000, "idiom": 6000, "ownership": 6000}
BANDS_LEGACY_NOTE = "噪声带一律读 bands.json（PREREG §4.1）"
CACHE = HERE / "cache"
SHARED = ROOT / "experiments" / "capability_map" / "cache"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact(d: dict, card: str) -> float:
    return d["end"][card]["exact_match"]


def bg_fraction(card: str, seed: int) -> float:
    """盲猜下界：永远输出"无切片"的 exact = 背景句占比（PREREG 空测试 4）。"""
    n = SAMPLES[card]
    for p in (CACHE / f"{card}_{n}.pkl", SHARED / f"{card}_{n}_20240927.pkl"):
        if p.exists():
            data = pickle.loads(p.read_bytes())
            break
    else:
        return float("nan")
    import random
    work = list(data)
    random.Random(seed).shuffle(work)
    nv = max(200, len(work) // 10)
    val = work[:nv]
    return sum(1 for s in val if not s["spans"]) / max(1, len(val))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-dir", default=str(HERE / "cards"))
    ap.add_argument("--bands", default=str(HERE / "bands.json"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args(argv)
    cards_dir = Path(args.cards_dir)
    out_path = Path(args.out)

    bands = load(Path(args.bands))["bands"]
    j3 = {s: load(ROOT / "experiments" / "core_keep" / "cards"
                  / f"J3_s{s}_metrics.json")["metrics"]["negation"]["exact_match"]
          for s in SEEDS}
    ctrl_neg = j3

    res: dict = {"bands": bands, "band_pt": {k: round(v * 100, 2) for k, v in bands.items()},
                 "seeds": {}, "p3": {}, "bg_frac": {}}
    missing, gate_fail = [], []
    verdict_p1, verdict_p2 = True, True

    for s in SEEDS:
        res["bg_frac"][str(s)] = {c: round(bg_fraction(c, s), 4) for c in SAMPLES}
        gp = cards_dir / f"gate0_s{s}.json"
        if not gp.exists():
            missing.append(gp.name)
            verdict_p1 = verdict_p2 = False
            res["seeds"][str(s)] = {"step0_exact": None, "gate_ok_step0": False,
                                    "steps": {}}
            continue
        g = load(gp)
        if not g["gate"]["ok"]:
            gate_fail.append(f"gate0_s{s}")
        # 累计基线：四张老卡 = 步 0；后来加入的卡 = 它第一次训完那步的末值
        base = {c: g["end"][c]["exact_match"] for c in OLD}
        row = {"step0_exact": base, "gate_ok_step0": g["gate"]["ok"], "steps": {}}

        for step in (1, 2, 3):
            p = cards_dir / f"step{step}_s{s}.json"
            if not p.exists():
                missing.append(p.name)
                verdict_p1 = verdict_p2 = False
                continue
            m = load(p)
            if not m["gate"]["ok"]:
                gate_fail.append(f"step{step}_s{s}")
                verdict_p1 = verdict_p2 = False
            cards = STEP_CARDS[step]
            st = {c: m["start"][c]["exact_match"] for c in m["frozen"]}
            en = {c: m["end"][c]["exact_match"] for c in cards}
            base.setdefault(NEW[step], en[NEW[step]])

            # ---- P1：逐卡 Δ ≥ −band（单侧）+ 双侧次读法 ----
            p1_rows, p1_ok, p1_ok_two = {}, True, True
            for c in m["frozen"]:
                d = en[c] - st[c]
                b = bands[c]
                one = d >= -b
                two = abs(d) <= b
                p1_ok &= one
                p1_ok_two &= two
                cum = (en[c] - base[c]) if c in base else None
                p1_rows[c] = {"start": st[c], "end": en[c], "delta": d,
                              "delta_pt": round(d * 100, 3), "band": b,
                              "pass_one_side": one, "pass_two_side": two,
                              "delta_vs_step0": cum,
                              "delta_vs_step0_pt": (round(cum * 100, 3)
                                                    if cum is not None else None),
                              "floor": res["bg_frac"][str(s)][c],
                              "on_floor": en[c] <= res["bg_frac"][str(s)][c] + 0.02}
            verdict_p1 &= p1_ok

            # ---- P2：新卡 ≥ 0.9 × 对照 ----
            new = NEW[step]
            cp = cards_dir / f"ctrl_{new}_s{s}.json"
            if new == "negation":
                ctrl = ctrl_neg[s]
            elif cp.exists():
                ctrl = exact(load(cp), new)
            else:
                ctrl = None
            if ctrl:
                ratio = en[new] / ctrl
                p2 = ratio >= 0.90
            else:
                ratio, p2 = float("nan"), False
                missing.append(cp.name)
            verdict_p2 &= p2

            row["steps"][str(step)] = {
                "new": new, "exact_start": st, "exact_end": en, "p1": p1_rows,
                "p1_pass": p1_ok, "p1_pass_two_side": p1_ok_two,
                "p2": {"new": new, "exact": en[new], "ctrl": ctrl,
                       "ratio": ratio, "ratio_pct": round(ratio * 100, 1),
                       "line": 0.90, "pass": p2,
                       "degenerate": new == "negation"},
                "timing": m["timing"], "drift": m["drift"],
                "params_trainable": m["params"]["trainable_total"],
            }
        res["seeds"][str(s)] = row

    # ---------------- P3 ----------------
    for s in SEEDS:
        row = res["seeds"][str(s)]["steps"]
        d = [row[str(i)]["drift"]["vs_base_end"] if str(i) in row else None
             for i in (1, 2, 3)]
        delta = [d[0] if d[0] is not None else None,
                 (d[1] - d[0]) if (d[0] is not None and d[1] is not None) else None,
                 (d[2] - d[1]) if (d[1] is not None and d[2] is not None) else None]
        timing = {i: row[str(i)]["timing"] for i in (1, 2, 3) if str(i) in row}
        sec_growth = {i: timing[i]["sec_per_step_steady"] / timing[1]["sec_per_step_steady"]
                      for i in timing if i != 1} if 1 in timing else {}
        mem_growth = {i: timing[i]["peak_mem_mb"] / timing[1]["peak_mem_mb"]
                      for i in timing if i != 1} if 1 in timing else {}
        res["p3"][str(s)] = {
            "drift_vs_base": d, "delta": delta,
            "sec_per_step": {i: timing[i]["sec_per_step_steady"] for i in timing},
            "peak_mb": {i: timing[i]["peak_mem_mb"] for i in timing},
            "sec_growth_vs_step1": sec_growth, "mem_growth_vs_step1": mem_growth,
        }

    # 线性 vs 加速（PREREG §4 P3：δ2 vs δ3，两 seed 同号）
    try:
        d2 = [res["p3"][str(s)]["delta"][1] for s in SEEDS]
        d3 = [res["p3"][str(s)]["delta"][2] for s in SEEDS]
        if any(x is None or y is None for x, y in zip(d2, d3)):
            shape = "证据不足（步 2/3 缺失）"
        else:
            up = all(y > x for x, y in zip(d2, d3))
            down = all(y < x for x, y in zip(d2, d3))
            shape = "加速" if up else ("减速/收敛" if down else "混合")
    except (KeyError, TypeError, IndexError):
        shape = "证据不足"
    res["p3_shape"] = shape

    # 代价失控（PREREG：相对步 1 增长 >20%，两 seed 同号）
    # PREREG §4 P3 写死：步时 **或** 峰值显存，某一步相对步 1 增长 >20% 且两 seed 同号
    cost_bad, cost_detail = False, {}
    for kind in ("sec_growth_vs_step1", "mem_growth_vs_step1"):
        for i in (2, 3):
            vals = [res["p3"][str(s)][kind].get(i) for s in SEEDS]
            if all(v is not None and v > 1.20 for v in vals):
                cost_bad = True
                cost_detail[f"{kind}@step{i}"] = [round(v, 4) for v in vals]
    res["cost_overrun"] = bool(cost_bad)
    res["cost_detail"] = cost_detail
    # 双侧读法（§14：字面 |Δ| ≤ 带，与单侧并列报）
    res["p1_two_all"] = all(
        res["seeds"][str(s)]["steps"].get(str(step), {}).get("p1_pass_two_side", False)
        for s in SEEDS for step in (1, 2, 3)
        if str(step) in res["seeds"][str(s)]["steps"])

    # ---------------- 判定 ----------------
    if missing or gate_fail:
        verdict = "证据不足"
        why = {"missing": missing, "gate_fail": gate_fail}
    elif verdict_p1 and verdict_p2:
        verdict = "流程可反复用"
        why = {}
    else:
        verdict = "不可反复用"
        why = {"p1": verdict_p1, "p2": verdict_p2}
        # 指出第几步、哪张卡先崩
        first = []
        for s in SEEDS:
            for step in (1, 2, 3):
                st = res["seeds"][str(s)]["steps"].get(str(step))
                if not st:
                    continue
                if not st["p1_pass"]:
                    bad = [c for c, r in st["p1"].items() if not r["pass_one_side"]]
                    first.append(f"s{s} step{step} P1 崩卡={bad}")
                if not st["p2"]["pass"]:
                    first.append(f"s{s} step{step} P2 崩卡={st['p2']['new']}"
                                 f" (ratio={st['p2']['ratio_pct']}%)")
        why["first_fail"] = first

    res["verdict"] = verdict
    res["why"] = why
    res["missing"] = missing
    res["gate_fail"] = gate_fail
    res["p1_all"] = bool(verdict_p1)
    res["p2_all"] = bool(verdict_p2)

    (out_path).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[save] {out_path}")

    # ---------------- 主表 ----------------
    print("\n=== 主表 1：每步 × 每张已存在卡 × 2 seed（exact；Δ=该步相对起点）===")
    print("| seed | 步 | 卡 | 起点 | 末态 | Δ(pt) | 带(pt) | P1单侧 | 累计Δ vs步0(pt) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        for step in (1, 2, 3):
            st = res["seeds"][str(s)]["steps"].get(str(step))
            if not st:
                continue
            for c, r in st["p1"].items():
                cum = "—" if r["delta_vs_step0_pt"] is None else f"{r['delta_vs_step0_pt']:+.2f}"
                print(f"| {s} | {step} | {c} | {r['start']:.4f} | {r['end']:.4f} | "
                      f"{r['delta_pt']:+.2f} | {r['band']*100:.2f} | "
                      f"{'✅' if r['pass_one_side'] else '❌'} | {cum} |")

    print("\n=== 主表 2：新卡达标率（P2，线 = 对照 × 0.90）===")
    print("| seed | 步 | 新卡 | exact | 对照 | 比值 | 0.90 线 | P2 | 备注 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        for step in (1, 2, 3):
            st = res["seeds"][str(s)]["steps"].get(str(step))
            if not st:
                continue
            p = st["p2"]
            note = "同构/恒等式" if p["degenerate"] else "对照=单加卡"
            ctrl = p["ctrl"] if p["ctrl"] is not None else float("nan")
            print(f"| {s} | {step} | {p['new']} | {p['exact']:.4f} | {ctrl:.4f} | "
                  f"{p['ratio_pct']:.1f}% | {ctrl*0.9:.4f} | "
                  f"{'✅' if p['pass'] else '❌'} | {note} |")

    print("\n=== 主表 3：P3 代价与核漂移 ===")
    print("| seed | 步 | 稳态步时(s) | 峰值显存(MB) | d(base→末态) | δ 本步增量 | 可训参数 |")
    print("|---|---|---|---|---|---|---|")
    for s in SEEDS:
        p3 = res["p3"][str(s)]
        for i in (1, 2, 3):
            st = res["seeds"][str(s)]["steps"].get(str(i))
            if not st:
                continue
            t = st["timing"]
            dl = p3["delta"][i - 1]
            print(f"| {s} | {i} | {t['sec_per_step_steady']:.3f} | "
                  f"{t['peak_mem_mb']:.0f} | {p3['drift_vs_base'][i-1]:.4f} | "
                  f"{'—' if dl is None else f'{dl:.4f}'} | {st['params_trainable']:,} |")

    print(f"\n漂移形状（δ2 vs δ3，两 seed 同号）：{res['p3_shape']}")
    print(f"代价失控（步时或峰值显存 >20%×步1，两 seed 同号）：{res['cost_overrun']} "
          f"{json.dumps(res.get('cost_detail', {}), ensure_ascii=False)}")
    print(f"P1 双侧读法（|Δ|≤带，§14 次读法）：{res['p1_two_all']}")
    print(f"P1={res['p1_all']} P2={res['p2_all']} gate_fail={gate_fail} missing={missing}")
    print(f"\n### 判定：{verdict}")
    if why:
        print(json.dumps(why, ensure_ascii=False))
    print("SUMMARIZE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
