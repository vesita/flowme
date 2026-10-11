#!/usr/bin/env python3
"""S86m · H2（[`PREREG_GENOBJ.md`](../experiments/meme_anchor/PREREG_GENOBJ.md)）判据表：**tf vs roll 的配对判词**。

    .venv/bin/python stages/86m_genobj_verdict.py              # 读 logs/86g_result_*.json 出判词
    .venv/bin/python stages/86m_genobj_verdict.py --selfcheck  # R48：用伪造结果逐档驱动**真实判词函数**

判据（跑前写死，见 PREREG §3/§4）：
  **G1** 宏平均自由生成 EM 配对 `Δ ≥ 2×SE_paired` 且 4/4 同号 · **G1-ext** `|Δ|∈[1SE,2SE)` ⇒ 需扩到 n=6
  **G2** 重构配对掉 ≤ `max(1pp, 2×SE_pair)` · **G3** 衔接 Δ / AUROC 配对掉 ≤ `max(0.01, 2×SE_pair)`
  **G4** 读数完备（4 seed × 2 臂）· **G5** 报 `Δ_min = 2×SE_paired` · **G6** 饱和（读 snaps）
  判决五分支：一成立 / 二代价过大 / 三未测出 / 四有害 / 五作废。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/vesita/coding/my/flowme")
LOGS = ROOT / "logs"
SEEDS = (1234, 5678, 9012, 3456)
STEPS = 8000
G2_FLOOR = 0.01          # 1pp
G3_FLOOR = 0.01


# ---------------------------------------------------------------- 读数装配
def _g(r: dict, *path):
    x = r
    for k in path:
        x = x[k]
    return float(x)


def load_runs() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(LOGS / "86g_result_*.json"))):
        try:
            out.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def load_reeval(path: str) -> dict:
    """★ 统一口径读数（`logs/genobj_reeval.json`）：把 8 个 ckpt 用**同一套评测**（`eval()` + `chunk=64` + `n=1000`）重测的结果。

    为什么要它：原 run 的 JSON 里，**6 个是旧口径**（评测未切 `eval()`、chunk=256）、**2 个是重跑的新口径** ⇒ 直接比较会混口径。
    H2 的**官方判词以本文件为准**（`REPORT_GENOBJ §7`）。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    tab: dict = {}
    for k, v in d.items():
        o, s = k.split("#")
        tab[(o, int(s))] = dict(
            gen=dict(v["gen"]), final=dict(v["final"]),
            metrics={"bridge": {"pooled": {"delta": v["bridge"]["delta"]},
                                "auroc_centered": v["bridge"]["auroc"]}},
            steps=STEPS, seed=int(s), obj=o)
    return tab


def pick(runs: list[dict], steps: int = STEPS) -> dict:
    """筛出本试验的两臂：bridge=parser ∧ frozen ∧ 载入 α0.5 输入卡 ∧ obj∈{tf,roll} ∧ steps 匹配。"""
    tab: dict = {}
    for r in runs:
        if r.get("bridge") != "parser":
            continue
        if not r.get("frozen"):
            continue
        if "_a0.5_" not in str(r.get("init_inputs", "")):
            continue
        if r.get("obj") not in ("tf", "roll"):
            continue
        if int(r.get("steps", 0)) != steps:
            continue
        tab[(r["obj"], int(r["seed"]))] = r
    return tab


def pair_delta(tab: dict, *path) -> list[float]:
    """同 seed 配对 Δ = roll − tf（缺任一侧该 seed 直接不出现，由 G4 兜）。"""
    ds = []
    for s in SEEDS:
        a, b = tab.get(("tf", s)), tab.get(("roll", s))
        if a is None or b is None:
            continue
        ds.append(_g(b, *path) - _g(a, *path))
    return ds


def mean_se(ds: list[float]) -> tuple[float, float]:
    k = len(ds)
    if k == 0:
        return float("nan"), float("nan")
    m = sum(ds) / k
    if k < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in ds) / (k - 1)
    return m, math.sqrt(var) / math.sqrt(k)


# ---------------------------------------------------------------- 判词（R48 驱动的就是它）
def verdict(tab: dict) -> dict:
    missing = [f"{o}#{s}" for o in ("tf", "roll") for s in SEEDS if (o, s) not in tab]
    if missing:                                                    # G4
        return dict(branch="五", text=f"【判词作废（五）】读数不完备 G4：缺 {missing}")

    dz = pair_delta(tab, "gen", "gen_em_zz")
    de = pair_delta(tab, "gen", "gen_em_ee")
    dmean = [(a + b) / 2 for a, b in zip(dz, de)]                  # 宏平均（G1 主指标）
    m, se = mean_se(dmean)
    dmin = 2 * se

    dcz = pair_delta(tab, "final", "char_zz")
    dce = pair_delta(tab, "final", "char_ee")
    dbr = pair_delta(tab, "metrics", "bridge", "pooled", "delta")
    dau = pair_delta(tab, "metrics", "bridge", "auroc_centered")

    def _m(ds):
        return sum(ds) / len(ds) if ds else float("nan")

    thr_g2 = max(G2_FLOOR, 2 * mean_se(dcz)[1])                    # 噪声带口径（不再拍死 2pp）
    thr_g3d = max(G3_FLOOR, 2 * mean_se(dbr)[1])
    thr_g3a = max(G3_FLOOR, 2 * mean_se(dau)[1])

    g1 = bool(m >= 2 * se and all(d > 0 for d in dmean))
    in_band = bool(se == se and abs(m) >= se and abs(m) < 2 * se)  # G1-ext：功效不足带
    g2 = bool(min(_m(dcz), _m(dce)) >= -thr_g2)
    g3 = bool(_m(dbr) >= -thr_g3d and _m(dau) >= -thr_g3a)
    neg = bool(m <= -2 * se and all(d < 0 for d in dmean))

    if g1 and g2 and g3:
        branch, text = "一", "【成立（一）】生成缺口是目标问题 ⇒ rollout 项进 v0 配方；再进 H1/H6"
    elif g1:
        branch, text = "二", (f"【代价过大（二）】rollout 提升自由生成但破非劣门："
                              f"G2 {'过' if g2 else '破'}（阈值 {thr_g2*100:.2f}pp）/ G3 {'过' if g3 else '破'}"
                              f"（阈值 Δ {thr_g3d:.4f} / AUROC {thr_g3a:.4f}）⇒ 记互斥，只许调权重或换形态（须新预注册）")
    elif neg:
        branch, text = "四", "【有害（四）】配对 Δ ≤ −2×SE 且 4/4 为负 ⇒ rollout 在本规模下有害，写进已判死"
    else:
        ext = "★ 落在 [1SE,2SE) 带内 ⇒ 按 G1-ext **补 2 seed 扩到 n=6** 再判" if in_band else ""
        branch, text = "三", (f"【未测出（三）】Δ_mean={m*100:+.2f}pp < Δ_min={dmin*100:.2f}pp ⇒ "
                              f"只能写「该口径下未测出差异」；不纠缠（§0.2），转 H6。{ext}")

    return dict(branch=branch, text=text, m=m, se=se, dmin=dmin, g1=g1, g2=g2, g3=g3,
                in_band=in_band, dz=dz, de=de, dmean=dmean, dcz=dcz, dce=dce, dbr=dbr, dau=dau,
                thr_g2=thr_g2, thr_g3d=thr_g3d, thr_g3a=thr_g3a, missing=missing)


# ---------------------------------------------------------------- 报表 / 饱和
def report(tab: dict) -> None:
    print(f"[GENOBJ] 两臂 run：tf={sum(1 for k in tab if k[0]=='tf')} / roll={sum(1 for k in tab if k[0]=='roll')}"
          f"（seed 需要 {list(SEEDS)}；steps={STEPS}）")
    if not tab:
        print("[GENOBJ] 没有筛出任何 run ⇒ 批还没跑完或 tag 不对")
        return
    print("\n| seed | obj | 生成EM 中→中 / 英→英 | 生成 n | 重构 char 中→中 / 英→英 | 衔接Δ | AUROC | 停止 |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        for o in ("tf", "roll"):
            r = tab.get((o, s))
            if r is None:
                print(f"| {s} | {o} | — | — | — | — | — | — |")
                continue
            print(f"| {s} | {o} | {r['gen']['gen_em_zz']*100:.1f}% / {r['gen']['gen_em_ee']*100:.1f}% "
                  f"| {r['gen'].get('gen_n')} | {r['final']['char_zz']*100:.2f}% / {r['final']['char_ee']*100:.2f}% "
                  f"| {_g(r,'metrics','bridge','pooled','delta'):+.4f} | {_g(r,'metrics','bridge','auroc_centered'):.4f} "
                  f"| {r.get('stopped')} |")
    v = verdict(tab)
    if v.get("missing"):
        print(f"\n[GENOBJ-VERDICT] {v['text']}")
        return
    print(f"\n[G1 ] 宏平均配对 Δ_mean={v['m']*100:+.2f}pp，SE_paired={v['se']*100:.2f}pp，"
          f"Δ_min={v['dmin']*100:.2f}pp，逐 seed Δ={[round(x*100,2) for x in v['dmean']]}")
    print(f"[G5 ] Δ_min = 2×SE_paired = {v['dmin']*100:.2f}pp（|Δ|<Δ_min ⇒ 只许写「未测出差异」）")
    print(f"[G2 ] 重构配对掉幅：中→中 {sum(v['dcz'])/len(v['dcz'])*100:+.2f}pp / 英→英 {sum(v['dce'])/len(v['dce'])*100:+.2f}pp"
          f"（门槛 {v['thr_g2']*100:.2f}pp）")
    print(f"[G3 ] 衔接 Δ 掉幅 {sum(v['dbr'])/len(v['dbr']):+.4f}（门槛 {v['thr_g3d']:.4f}）；"
          f"AUROC 掉幅 {sum(v['dau'])/len(v['dau']):+.4f}（门槛 {v['thr_g3a']:.4f}）")
    print(f"\n[GENOBJ-VERDICT] {v['text']}")
    for o in ("tf", "roll"):
        for s in SEEDS:
            r = tab.get((o, s))
            if r and len(r.get("snaps", [])) >= 2:
                tot = [sum(x for k, x in sn.items() if k.startswith("ce_")) for sn in r["snaps"]]
                print(f"[G6] {o}#{s} 末两格 CE 和增量 {tot[-2]-tot[-1]:+.4f}（<0.01 视为平；stopped={r.get('stopped')}）")


# ---------------------------------------------------------------- R48 假结果自检
def _fake(deltas, *, char_drop=0.0, br_drop=0.0, au_drop=0.0, drop_seed=None) -> dict:
    """造 4 seed × 2 臂的假结果：gen EM 按 `deltas`（宏平均两侧同值），其余按给定掉幅。"""
    tab = {}
    for i, s in enumerate(SEEDS):
        if drop_seed is not None and s == drop_seed:
            continue
        for o in ("tf", "roll"):
            d = deltas[i] if o == "roll" else 0.0
            tab[(o, s)] = dict(
                gen={"gen_em_zz": 0.65 + d, "gen_em_ee": 0.65 + d, "gen_n": 1000},
                final={"char_zz": 0.92 - (char_drop if o == "roll" else 0.0),
                       "char_ee": 0.96 - (char_drop if o == "roll" else 0.0)},
                metrics={"bridge": {"pooled": {"delta": 0.60 - (br_drop if o == "roll" else 0.0)},
                                   "auroc_centered": 0.99 - (au_drop if o == "roll" else 0.0)}},
                snaps=[], steps=STEPS)
    return tab


def selfcheck() -> None:
    cases = [
        ("一（成立）", _fake([0.10, 0.11, 0.09, 0.10]), "一", False),
        ("二（代价过大）", _fake([0.10, 0.11, 0.09, 0.10], char_drop=0.05), "二", False),
        ("三（未测出·带内 G1-ext）", _fake([0.008, -0.002, 0.006, 0.004]), "三", True),
        ("四（有害）", _fake([-0.10, -0.11, -0.09, -0.10]), "四", False),
        ("五（作废·缺一 run）", _fake([0.10, 0.11, 0.09, 0.10], drop_seed=3456), "五", False),
    ]
    ok = True
    for name, tab, expect, want_band in cases:
        v = verdict(tab)
        good = v["branch"] == expect and (bool(v.get("in_band")) == want_band)
        ok = ok and good
        print(f"[{'✅' if good else '❌'}] {name} → 分支 {v['branch']}（预期 {expect}）"
              f" in_band={v.get('in_band')}\n     {v['text']}")
    print(f"\n[GENOBJ-SELFCHECK] {'OK' if ok else 'FAIL'}（五档 + G1-ext 带内）")
    if not ok:
        sys.exit(9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--from-reeval", default="", metavar="JSON",
                    help="读统一口径重测读数（logs/genobj_reeval.json）；默认读 logs/86g_result_*.json")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.from_reeval:
        print(f"[GENOBJ] 数据源 = 统一口径重测 {args.from_reeval}")
        report(load_reeval(args.from_reeval))
        return
    report(pick(load_runs(), steps=args.steps))


if __name__ == "__main__":
    main()
