#!/usr/bin/env python3
"""S86b · M1 **跑后附加对照**（不改判定、不改在跑代码）：长度匹配对照 + 内容探针。

为什么需要：
1. **长度匹配对照** —— 图片是同一句渲染出来的 ⇒ 图片模因天然与"句长/版式"相关。
   随机配对（不同长度）当对照时，Δ>0 可能只反映"长度相近"。⇒ 用**同长度、异内容**当负样本重算 Δ；
   若同长度下 Δ 仍 ≥2SE 且 ≥0.01，才能说"跨模态相似不是句长造成的"。
2. **内容探针** —— 从模因线性预测「句长 / 首字 / 末字」：若只能预测句长，说明模因没承载内容。

只读 `logs/86_ckpt_*.pt` 与数据缓存；只写 `logs/86_extra.json`。**不重训、不改判据。**

    python stages/86b_m1_extra.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/vesita/coding/my/flowme")
HERE = ROOT / "experiments" / "meme_anchor"
sys.path.insert(0, str(HERE))
import measure as MS   # noqa: E402

_spec = importlib.util.spec_from_file_location("m1driver", ROOT / "stages" / "86_meme_anchor.py")
DRV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRV)

LOGS = ROOT / "logs"
TOPK = 100


def log(m: str) -> None:
    print(m, flush=True)


def length_matched_delta(S: torch.Tensor, lens: np.ndarray) -> dict:
    """Δ_len_i = S[i,i] − mean_{j: 同长度, j≠i} S[i,j]（配对 SE）。"""
    groups: dict[int, list[int]] = {}
    for i, L in enumerate(lens):
        groups.setdefault(int(L), []).append(i)
    d = []
    for i, L in enumerate(lens):
        js = [j for j in groups[int(L)] if j != i]
        if not js:
            continue
        d.append(float(S[i, i] - S[i, js].mean()))
    a = np.asarray(d, float)
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")
    return dict(n=int(a.size), delta=float(a.mean()), delta_se=se,
                pass2se=bool(se > 0 and a.mean() >= 2 * se),
                pass_floor=bool(a.mean() >= 0.01),
                pass_final=bool(se > 0 and a.mean() >= 2 * se and a.mean() >= 0.01),
                n_per_len=float(np.mean([len(v) for v in groups.values()])))


def _ridge(X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0):
    """标准化 X、**中心化 Y（带截距）**后岭回归；返回 (Xs, W, ymu)。

    ★ 自检教训（2026-10-11）：早先版本 X 标准化了、Y 没中心化 ⇒ 预测均值恒为 0 ⇒
    连"直接用含句长的特征预测句长"都给出 R²=−5.1（应为 ≈1）⇒ **探针坏了，不是数据没信号**。
    """
    ntr = X.size(0) // 2
    xmu, xsd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xs = (X - xmu) / xsd
    ymu = Y[:ntr].mean(0)
    Yc = Y - ymu
    W = torch.linalg.solve(Xs[:ntr].t() @ Xs[:ntr] + lam * torch.eye(X.size(1)),
                           Xs[:ntr].t() @ Yc[:ntr])
    return Xs, W, ymu


def ridge_probe(X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0) -> float:
    """前一半拟合、后一半评估；Y 为 one-hot ⇒ 返回 top-1 准确率。"""
    Xs, W, ymu = _ridge(X, Y, lam)
    ntr = X.size(0) // 2
    pred = (Xs[ntr:] @ W + ymu).argmax(1)
    return float((pred == Y[ntr:].argmax(1)).float().mean())


def ridge_r2(X: torch.Tensor, y: torch.Tensor, lam: float = 1.0) -> float:
    Xs, W, ymu = _ridge(X, y.unsqueeze(1) if y.dim() == 1 else y, lam)
    ntr = X.size(0) // 2
    yv = y if y.dim() == 1 else y.squeeze(1)
    pred = (Xs[ntr:] @ W + ymu).squeeze(1)
    ss_res = float(((yv[ntr:] - pred) ** 2).sum())
    ss_tot = float(((yv[ntr:] - yv[ntr:].mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def dist_report(S: torch.Tensor) -> dict:
    """§9 要求的「完整分布」：同内容/随机配对/Δ 的分位数 + 均匀口径的 AUROC 与最优阈值。"""
    S = S.float()
    N = S.size(0)
    diag = torch.diagonal(S)
    off = S[~torch.eye(N, dtype=torch.bool)]
    rowmean = (S.sum(1) - diag) / max(N - 1, 1)
    delta = diag - rowmean
    q = [5, 25, 50, 75, 95]
    mask = ~torch.eye(N, dtype=torch.bool)
    centered = S - rowmean.unsqueeze(1)
    return dict(
        same_q={f"p{x}": float(torch.quantile(diag, x / 100)) for x in q},
        rand_q={f"p{x}": float(torch.quantile(off, x / 100)) for x in q},
        delta_q={f"p{x}": float(torch.quantile(delta, x / 100)) for x in q},
        same_mean=float(diag.mean()), rand_mean=float(off.mean()), delta_mean=float(delta.mean()),
        auroc=MS.auroc_rank(diag, off), thr_raw=MS.youden_threshold(diag, off),
        thr_centered=MS.youden_threshold(centered.diagonal(), centered[mask]))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[EXTRA] device={device}")
    pack = DRV.get_pack()
    sents = pack["test"]
    lens = np.asarray([len(s) for s in sents])
    first = [s[0] for s in sents]
    last = [s[-1] for s in sents]

    top_first = [c for c, _ in Counter(first).most_common(TOPK)]
    top_last = [c for c, _ in Counter(last).most_common(TOPK)]
    i_f = {c: k for k, c in enumerate(top_first)}
    i_l = {c: k for k, c in enumerate(top_last)}
    keep_f = [k for k, c in enumerate(first) if c in i_f]
    keep_l = [k for k, c in enumerate(last) if c in i_l]

    def onehot(labels, idx) -> torch.Tensor:
        """只给**在 top-K 里**的标签打 one-hot（其余行全 0，调用处用 keep_* 过滤）。"""
        Y = torch.zeros(len(labels), len(idx))
        for r, c in enumerate(labels):
            k = idx.get(c)
            if k is not None:
                Y[r, k] = 1.0
        return Y

    Yf = onehot(first, i_f)
    Yl = onehot(last, i_l)
    chance_f = max(Counter(first).values()) / len(first)
    chance_l = max(Counter(last).values()) / len(last)
    log(f"[EXTRA] test={len(sents)} 句长 {lens.min()}..{lens.max()} 中位 {int(np.median(lens))}；"
        f"首字类 {len(i_f)}（众数基线 {chance_f*100:.2f}%）末字类 {len(i_l)}（{chance_l*100:.2f}%）")

    out = {}
    for ck in sorted(LOGS.glob("86_ckpt_*.pt")):
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        lam, seed, arch = blob["lam"], blob["seed"], blob["vision_arch"]
        model = DRV.MD.build(seed, arch)
        model.load_state_dict(blob["state"])
        model.to(device)
        ev = DRV.eval_model(model, pack["test_ids"], pack["test_img"], device)
        pT, pI = MS.pooled(ev["m_T"]), MS.pooled(ev["m_I"])
        S = MS.cosmat(pT, pI)

        lenm = length_matched_delta(S, lens)
        rnd = MS.pair_stats(S)
        row = dict(lam=lam, seed=seed, arch=arch, dist=dist_report(S),
                   delta_random=dict(delta=rnd["delta"], se=rnd["delta_se"],
                                     pass_final=rnd["pass_final"]),
                   delta_len_matched=lenm,
                   probe_length_r2=dict(T=ridge_r2(pT, torch.tensor(lens, dtype=torch.float32)),
                                        I=ridge_r2(pI, torch.tensor(lens, dtype=torch.float32))),
                   probe_first_char=dict(T=ridge_probe(pT[keep_f], Yf[keep_f]),
                                         I=ridge_probe(pI[keep_f], Yf[keep_f]),
                                         chance=chance_f),
                   probe_last_char=dict(T=ridge_probe(pT[keep_l], Yl[keep_l]),
                                        I=ridge_probe(pI[keep_l], Yl[keep_l]),
                                        chance=chance_l))
        out[f"l{lam}_s{seed}"] = row
        log(f"[EXTRA] λ={lam} seed={seed} | Δ随机={rnd['delta']:+.4f}±{rnd['delta_se']:.4f} "
            f"(过={rnd['pass_final']}) | **Δ长度匹配={lenm['delta']:+.4f}±{lenm['delta_se']:.4f}** "
            f"(过={lenm['pass_final']}, 同长度平均 {lenm['n_per_len']:.1f} 个伙伴) | "
            f"句长R² T={row['probe_length_r2']['T']:+.3f} I={row['probe_length_r2']['I']:+.3f} | "
            f"首字 T={row['probe_first_char']['T']*100:.1f}% I={row['probe_first_char']['I']*100:.1f}% "
            f"(基线 {chance_f*100:.2f}%) | 末字 T={row['probe_last_char']['T']*100:.1f}% "
            f"I={row['probe_last_char']['I']*100:.1f}% (基线 {chance_l*100:.2f}%)"
            + ("  ★末字探针退化（test 集末字只有 1 类 ⇒ 无效）" if len(i_l) <= 1 else ""))
        dd = row["dist"]
        log(f"[EXTRA-DIST] λ={lam} s={seed} 同内容 cos p5/50/95="
            f"{dd['same_q']['p5']:.4f}/{dd['same_q']['p50']:.4f}/{dd['same_q']['p95']:.4f} | "
            f"随机 cos p5/50/95={dd['rand_q']['p5']:.4f}/{dd['rand_q']['p50']:.4f}/{dd['rand_q']['p95']:.4f} "
            f"| Δ p5/50/95={dd['delta_q']['p5']:+.4f}/{dd['delta_q']['p50']:+.4f}/{dd['delta_q']['p95']:+.4f} "
            f"| AUROC={dd['auroc']:.4f} 阈值(原始)={dd['thr_raw']['thr']:+.4f} "
            f"阈值(中心)={dd['thr_centered']['thr']:+.4f} TPR={dd['thr_centered']['tpr']:.3f} "
            f"FPR={dd['thr_centered']['fpr']:.3f}")
        del model

    (LOGS / "86_extra.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    log(f"[EXTRA] 已写 {LOGS/'86_extra.json'}")


if __name__ == "__main__":
    main()
