#!/usr/bin/env python3
"""S87 数据分析：把「调度效应」与「运行噪声」拆开算，并给 EM 的 McNemar 检验。

    python stages/87_analyze.py --steps 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/vesita/coding/my/flowme")
LOGS = ROOT / "logs"
sys.path.insert(0, str(ROOT / "stages"))


def load(sched: str, seed: int, steps: int) -> dict:
    return torch.load(LOGS / f"87_tok_{sched}_s{seed}_st{steps}.pt", weights_only=False)


def cluster_stats(d: torch.Tensor, mask: torch.Tensor, lens: list[int]) -> dict:
    per, off = [], 0
    for n in lens:
        sel = mask[off:off + n]
        if int(sel.sum()):
            per.append(float(d[off:off + n][sel].mean()))
        off += n
    s = torch.tensor(per, dtype=torch.float64)
    se = float(s.std(unbiased=True) / (s.numel() ** 0.5))
    return {"mean": float(s.mean()), "se": se, "dmin": 2 * se, "n_samples": int(s.numel()),
            "t": float(s.mean() / se) if se else float("nan")}


def mcnemar(a: list[int], b: list[int]) -> dict:
    """配对二值（逐样本 EM）的 McNemar 精确检验。b = 只有 A 对，c = 只有 B 对。"""
    b01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    c10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = b01 + c10
    if n == 0:
        p = 1.0
    else:
        from math import comb
        k = min(b01, c10)
        p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
        p = min(p, 1.0)
    return {"A对B错": b01, "A错B对": c10, "p": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    steps, bs = args.steps, args.bs

    import importlib
    V = importlib.import_module("87_sched_verdict")
    lens = V.chunk_lens(steps, bs)

    print("=" * 78)
    print(f"S87 数据分析 · steps={steps} · 语料=代用（multi_turn/shareai/agent dialogue）")
    print("=" * 78)

    # 1) 各臂读数
    rows = {}
    for seed in (1234, 5678):
        for sched in ("const", "cosine"):
            a = load(sched, seed, steps)
            m = a["pad_mask"]
            rows[(seed, sched)] = a
            print(f"[臂] seed={seed} {sched:6s} CE_t(有效token)={float(a['ce_tok_t'][m].mean()):.6f} "
                  f"CE_v={float(a['ce_tok_v'][m].mean()):.6f} "
                  f"EM={sum(a['exact_t'])/len(a['exact_t']):.4f} "
                  f"n_tok={int(m.sum())}")

    # 2) 主判据 + EM 检验 + A/A 对照
    aa = {}
    for seed in (1234, 5678):
        p = LOGS / f"87_aa_{'const'}_s{seed}_st{steps}.pt"
        if p.exists():
            aa[seed] = torch.load(p, weights_only=False)
        ap_ = LOGS / f"87_aa_tok_const_s{seed}_st{steps}.pt"
        if ap_.exists():
            t = torch.load(ap_, weights_only=False)
            aa[seed]["dce_vec"] = t["ce_tok_t_2"] - t["ce_tok_t_1"]
            aa[seed]["mask"] = t["pad_mask"]

    print("\n" + "-" * 78)
    print("主判据（ΔCE = cosine − const，按样本聚类）")
    print("-" * 78)
    print(f"{'seed':>6} {'ΔCE':>12} {'SE':>10} {'Δ_min':>10} {'t':>7} {'EM const→cos':>18} "
          f"{'McNemar p':>10} {'A/A ΔCE':>10} {'A/A EM差':>9}")
    results = {}
    for seed in (1234, 5678):
        a, b = rows[(seed, "const")], rows[(seed, "cosine")]
        d = b["ce_tok_t"].double() - a["ce_tok_t"].double()
        cs = cluster_stats(d, a["pad_mask"], lens)
        mc = mcnemar(a["exact_t"], b["exact_t"])
        em_a = sum(a["exact_t"]) / len(a["exact_t"])
        em_b = sum(b["exact_t"]) / len(b["exact_t"])
        aa_txt = ""
        if seed in aa and "dce_vec" in aa[seed]:
            aa_cs = cluster_stats(aa[seed]["dce_vec"].double(), aa[seed]["mask"], lens)
            aa_txt = f"{aa_cs['mean']:+.3e}"
            aa_em = aa[seed]["aa_dce_mean"] if "aa_dce_mean" in aa[seed] else float("nan")
        else:
            aa_em = float("nan")
        print(f"{seed:>6} {cs['mean']:>+12.4e} {cs['se']:>10.3e} {cs['dmin']:>10.3e} "
              f"{cs['t']:>7.1f} {em_a:>8.4f}→{em_b:<8.4f} {mc['p']:>10.3g} {aa_txt:>10} "
              f"{aa_em:>9.3e}")
        results[seed] = {"ce": cs, "mcnemar": mc, "em_const": em_a, "em_cosine": em_b}

    # 3) 判决
    per = {}
    for seed, r in results.items():
        per[seed] = {"missing": False, "dce": r["ce"]["mean"], "se": r["ce"]["se"],
                     "dmin": r["ce"]["dmin"], "n_samples": r["ce"]["n_samples"],
                     "em_const": r["em_const"], "em_cosine": r["em_cosine"], "em_floor": 0.0}
    v = V.verdict(per)
    print("\n" + "-" * 78)
    print(f"判词档位 = {v['档位']}")
    print(f"理由     = {v['理由']}")
    print("-" * 78)

    # 4) 效应 vs 噪声
    print("\n效应 vs 运行噪声（关键判读）")
    dces = [abs(r["ce"]["mean"]) for r in results.values()]
    aa_means = [abs(float(aa[s]["aa_dce_mean"])) for s in aa if "aa_dce_mean" in aa[s]]
    print(f"  两臂 |ΔCE|      = {dces[0]:.3e} / {dces[1]:.3e}")
    print(f"  A/A   |ΔCE|     = " + " / ".join(f"{x:.3e}" for x in aa_means) +
          "   ← 同配置两次运行的噪声")
    if aa_means:
        print(f"  效应/噪声 倍数  = {min(dces)/max(aa_means):.1f}×（最小效应 / 最大噪声）")
    print(f"  EM 变化         = {results[1234]['em_cosine']-results[1234]['em_const']:+.4f} / "
          f"{results[5678]['em_cosine']-results[5678]['em_const']:+.4f}（McNemar p="
          f"{results[1234]['mcnemar']['p']:.3g} / {results[5678]['mcnemar']['p']:.3g}）")

    (LOGS / f"87_analysis_st{steps}.json").write_text(
        json.dumps({"results": {str(k): {kk: vv for kk, vv in v2.items()} for k, v2 in results.items()},
                    "verdict": v["档位"], "verdict_reason": v["理由"],
                    "aa_dce_mean": aa_means}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print(f"\n[落盘] {LOGS}/87_analysis_st{steps}.json")


if __name__ == "__main__":
    main()
