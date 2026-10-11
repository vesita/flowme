#!/usr/bin/env python3
"""S86k · L3-m3d 2×2 判据表（`experiments/meme_anchor/PREREG_ALIGN_LOC.md` §2）。

三格（λ=0.3、infonce、都带适配卡）：
  A  输入**冻结** × 压力在**衔接处**（复用，L3-m3c）
  IX 输入**可训** × 压力在**衔接处**
  VIII 输入**可训** × 压力在**输入卡**
判据：B1 臂VIII 英→英自由生成 ≤ 臂A−5pp（代价复现）· B2 臂IX ≥ 臂A−5pp（位置隔离）。
用法（第二个参数可指向伪造结果的目录，用于 R48 预演）：
    .venv/bin/python stages/86k_align_loc.py [logs_dir]
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
LOGS = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/vesita/coding/my/flowme/logs")
MARGIN = 0.05

def g(r, *p):
    x = r
    for k in p: x = x[k]
    return float(x)

def mu(rs, *p):
    v = [g(r, *p) for r in rs]; n = len(v)
    return (sum(v)/n, (math.sqrt(sum((x-sum(v)/n)**2 for x in v)/(n-1))/math.sqrt(n)) if n > 1 else float("nan"), v)

def cell(r: dict) -> str:
    if r.get("loss_kind") != "infonce" or abs(float(r.get("lam", -1)) - 0.3) > 1e-9:
        return "忽略"
    if r.get("bridge") != "parser": return "忽略"
    if "_a0.5_" not in str(r.get("init_inputs", "")): return "忽略"
    ao, frz = r.get("align_on", "bridge"), bool(r.get("frozen"))
    if frz and ao == "bridge": return "A 冻结×衔接处"
    if not frz and ao == "bridge": return "IX 可训×衔接处"
    if not frz and ao == "input": return "VIII 可训×输入卡"
    return "忽略"

def load() -> dict:
    G: dict[str, list] = {}
    for f in sorted(glob.glob(str(LOGS / "86g_result_*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        k = cell(r)
        if k != "忽略": G.setdefault(k, []).append(r)
    return G

def verdict(G: dict) -> str:
    need = ("A 冻结×衔接处", "IX 可训×衔接处", "VIII 可训×输入卡")
    miss = [k for k in need if len(G.get(k, [])) < 2]
    if miss: return f"读数不完整：{ {k: len(G.get(k, [])) for k in need} }（缺 2 seed：{miss}）"
    pa, _, va = mu(G[need[0]], "gen", "gen_em_ee")
    pix, _, vix = mu(G[need[1]], "gen", "gen_em_ee")
    pviii, _, vviii = mu(G[need[2]], "gen", "gen_em_ee")
    b1 = pviii <= pa - MARGIN
    b2 = pix >= pa - MARGIN
    if b1 and b2: v = "✅ B1∧B2 ⇒ **「压力位置」是原因**（§0 的后验解释被实测证实）"
    elif b1 and not b2: v = "⚠ B1 过 / B2 不过 ⇒ **「输入卡可训」才是原因**（与压力位置无关）"
    else: v = "❌ B1 不过 ⇒ **L3-m2 的代价另有原因**（无适配卡/α/映射组合等），**不许沿用旧解释**，必须再查"
    return (f"[AL-VERDICT] {v}\n"
            f"  英→英自由生成 EM：臂A(冻结×衔接) {pa*100:.1f}% {[round(x*100,1) for x in va]} | "
            f"臂IX(可训×衔接) {pix*100:.1f}% {[round(x*100,1) for x in vix]} | "
            f"臂VIII(可训×输入) {pviii*100:.1f}% {[round(x*100,1) for x in vviii]}\n"
            f"  B1 臂VIII ≤ 臂A−5pp({(pa-MARGIN)*100:.1f}%) ⇒ {'✅ 代价复现' if b1 else '❌ 未复现'} | "
            f"B2 臂IX ≥ 臂A−5pp ⇒ {'✅ 位置隔离' if b2 else '❌ 也有代价'}")

def main() -> None:
    G = load()
    print(f"[AL] 分格：{ {k: len(v) for k, v in G.items()} }")
    if not G: print(verdict(G)); return
    print("\n| 格 | seed | 输入侧Δ | 衔接Δ | AUROC | 阈值 | 中→中 | 英→英 | 中→英 | 英→中 | 生成EM 中→中/英→英 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(G):
        for r in sorted(G[k], key=lambda x: x["seed"]):
            t = r["metrics"]["bridge"]["thr_centered"]
            print(f"| {k} | {r['seed']} | {g(r,'metrics','input','pooled','delta'):+.4f} | "
                  f"{g(r,'metrics','bridge','pooled','delta'):+.4f} | {g(r,'metrics','bridge','auroc_centered'):.4f} | "
                  f"{t['thr']:+.4f} | {r['final']['char_zz']*100:.1f}% | {r['final']['char_ee']*100:.1f}% | "
                  f"{r['final']['char_ze']*100:.1f}% | {r['final']['char_ez']*100:.1f}% | "
                  f"{r['gen']['gen_em_zz']*100:.1f}%/{r['gen']['gen_em_ee']*100:.1f}% |")
    print("\n" + verdict(G))

if __name__ == "__main__":
    main()
