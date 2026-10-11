#!/usr/bin/env python3
"""S86j · L3-m3c 判据表（`experiments/meme_anchor/PREREG_BRIDGE_ALIGN.md` §2）。

对照 = 臂 I（`bridge=parser` + `loss=none` + 冻结 α=0.5 输入）；候选 = 同结构 + **衔接处 infonce**（λ=0.1/0.3）。
    CUDA_VISIBLE_DEVICES="" .venv/bin/python stages/86j_bridge_align.py
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
LOGS = Path("/home/vesita/coding/my/flowme/logs")
MARGIN = 0.05

def g(r, *p):
    x = r
    for k in p: x = x[k]
    return float(x)

def mu(rs, *p):
    v = [g(r, *p) for r in rs]; n = len(v)
    return (sum(v)/n, (math.sqrt(sum((x-sum(v)/n)**2 for x in v)/(n-1))/math.sqrt(n)) if n > 1 else float("nan"), v)

def main() -> None:
    ctrl, cand = [], {}
    for f in sorted(glob.glob(str(LOGS / "86g_result_*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if r.get("bridge") != "parser" or not r.get("frozen"): continue
        if "_a0.5_" not in str(r.get("init_inputs", "")): continue
        if r.get("loss_kind") == "none" and not r.get("freeze_bridge"):
            ctrl.append(r)                                    # 臂 I
        elif r.get("loss_kind") == "infonce":
            cand.setdefault(float(r["lam"]), []).append(r)     # 臂 VI/VII
    if not ctrl or not cand:
        print(f"[BA] 读数不完整：对照 {len(ctrl)} / 候选 {sorted(cand)}"); return
    print(f"[BA] 对照臂 I: {len(ctrl)} run；候选: " + ", ".join(f"λ={k}×{len(v)}" for k, v in sorted(cand.items())))
    print("\n| 组 | seed | 输入侧Δ | **衔接Δ** | 长度匹配Δ | **AUROC** | 阈值 | TPR/FPR | 中→中 | 英→英 | 中→英 | 英→中 | 生成EM 中→中/英→英 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, rs in [("I 对照(无对齐项)", ctrl)] + [(f"λ={k}", v) for k, v in sorted(cand.items())]:
        for r in rs:
            t = r["metrics"]["bridge"]["thr_centered"]
            print(f"| {nm} | {r['seed']} | {g(r,'metrics','input','pooled','delta'):+.4f} | "
                  f"**{g(r,'metrics','bridge','pooled','delta'):+.4f}** | "
                  f"{g(r,'metrics','bridge','m8_length_matched','delta'):+.4f} | "
                  f"**{g(r,'metrics','bridge','auroc_centered'):.4f}** | {t['thr']:+.4f} | "
                  f"{t['tpr']:.3f}/{t['fpr']:.3f} | {r['final']['char_zz']*100:.1f}% | "
                  f"{r['final']['char_ee']*100:.1f}% | {r['final']['char_ze']*100:.1f}% | "
                  f"{r['final']['char_ez']*100:.1f}% | {r['gen']['gen_em_zz']*100:.1f}%/{r['gen']['gen_em_ee']*100:.1f}% |")
    for k, rs in sorted(cand.items()):
        rows = []
        for nm, path, kind in (
                ("A1a 衔接Δ ≥ 对照+2SE 且 2 seed 同号", ("metrics","bridge","pooled","delta"), "q"),
                ("A1b 长度匹配Δ 同上", ("metrics","bridge","m8_length_matched","delta"), "q"),
                ("A2 AUROC ≥ 对照+0.03", ("metrics","bridge","auroc_centered"), "au"),
                ("A3a 中→中 ≥ 对照−5pp", ("final","char_zz"), "m"),
                ("A3b 英→英 ≥ 对照−5pp", ("final","char_ee"), "m"),
                ("A4a 中→英 ≥ 对照−5pp", ("final","char_ze"), "m"),
                ("A4b 英→中 ≥ 对照−5pp", ("final","char_ez"), "m"),
                ("A5a 生成EM 中→中 ≥ 对照−5pp", ("gen","gen_em_zz"), "m"),
                ("A5b 生成EM 英→英 ≥ 对照−5pp", ("gen","gen_em_ee"), "m")):
            cm, cs, cv = mu(rs, *path)
            bm, _, _ = mu(ctrl, *path)
            ok = (cm >= bm + 0.03) if kind == "au" else ((cm >= bm + 2*cs and all(v > bm for v in cv)) if kind == "q" else (cm >= bm - MARGIN))
            rows.append((nm, bm, cm, ok, cv))
        print(f"\n[BA-GROUP] ==== λ={k} vs 臂 I ====")
        for nm, bm, cm, ok, cv in rows:
            print(f"[BA-ROW] {nm:<34} 对照 {bm:+.4f} → 候选 {cm:+.4f} {'✅' if ok else '❌'} 逐 seed={[round(v,4) for v in cv]}")
        print(f"[BA-VERDICT] λ={k}：" + ("✅ A1–A5 全过 ⇒ **衔接处施加对齐压力能同时拿到锐判据与不伤还原**"
              if all(r[3] for r in rows) else "❌ 未全过 ⇒ 按上面卡住的那条报读数"))

if __name__ == "__main__":
    main()
