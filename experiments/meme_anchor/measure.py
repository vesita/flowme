"""M1 度量层：相似度三口径 + 配对 Δ±SE + 正控 + 顺序自检 + 地板 + AUROC/阈值。

口径出处 `PREREG.md` §8（主口径 = mean-pool 余弦；并列逐槽 / CKA / 线性映射）、§9（阈值）、§6（自检）。

★ 一律在 CPU 上算（N=2000 的相似度矩阵 = 4M 元素，CPU 足够，且与设备无关 ⇒ 可复现）。
"""
from __future__ import annotations

import math

import numpy as np
import torch

I64 = torch.int64


# --------------------------------------------------------------------------- 基础口径
def _cpu(x) -> torch.Tensor:
    return x.detach().to("cpu").float() if isinstance(x, torch.Tensor) else torch.as_tensor(x).float()


def pooled(M: torch.Tensor) -> torch.Tensor:
    """mean-pool 到 [N, d]（主口径）。"""
    return _cpu(M).mean(dim=1)


def cosmat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return A @ B.t()


def slot_cos(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """逐槽余弦（同槽号对齐）⇒ [N]（每句的平均逐槽余弦）。"""
    A, B = _cpu(A), _cpu(B)
    a = A / A.norm(dim=2, keepdim=True).clamp_min(1e-12)
    b = B / B.norm(dim=2, keepdim=True).clamp_min(1e-12)
    return (a * b).sum(-1).mean(1)


def pair_stats(S: torch.Tensor) -> dict:
    """S[i,j] = sim(A_i, B_j)。返回同内容配对 / 随机配对（非对角全矩阵）与配对 Δ±SE。

    Δ_i = S[i,i] − mean_{j≠i} S[i,j]（逐句配对，PREREG §8 写死）。
    """
    S = _cpu(S)
    N = S.size(0)
    d = torch.diagonal(S)
    off = S.clone()
    off.fill_diagonal_(float("nan"))
    row_mean = torch.nanmean(off, dim=1)
    delta = d - row_mean
    off_flat = off[~torch.isnan(off)]

    def ms(x):
        n = x.numel()
        return (float(x.mean()), float(x.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan"))

    sm, sse = ms(d)
    rm, rse = ms(off_flat)
    dm, dse = ms(delta)
    # ★ 绝对效应量下限（PREREG 订正 3）：SE→0 时 "Δ≥2SE" 会平凡成立 ⇒ 必须同时要求 Δ 有量级。
    #   0.01 = S26 跨语言效应 +0.0524 的 1/5。
    floor = 0.01
    pass_stat = bool(dse > 0 and dm >= 2 * dse)
    return dict(n=N, same=sm, same_se=sse, rand=rm, rand_se=rse,
                delta=dm, delta_se=dse, t=(dm / dse if dse > 0 else float("nan")),
                pass2se=pass_stat, floor=floor, pass_floor=bool(dm >= floor),
                pass_final=bool(pass_stat and dm >= floor),
                unmeasured=bool(dse > 0 and abs(dm) <= dse),
                off_min=float(off_flat.min()), off_max=float(off_flat.max()),
                off_std=float(off_flat.std(unbiased=True)),
                nz_frac=float((delta.abs() > 1e-6).float().mean()),
                delta_i=delta)


def cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    X, Y = _cpu(X), _cpu(Y)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return float((X.t() @ Y).norm() ** 2 /
                 ((X.t() @ X).norm() * (Y.t() @ Y).norm()).clamp_min(1e-30))


def cka_perm_null(X: torch.Tensor, Y: torch.Tensor, nperm: int, seed: int) -> tuple[float, float]:
    g = torch.Generator().manual_seed(seed)
    vals = [cka_linear(X, Y[torch.randperm(Y.size(0), generator=g)]) for _ in range(nperm)]
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def linmap(A: torch.Tensor, B: torch.Tensor, lam: float = 1e-2,
           derange: bool = False) -> tuple[dict, dict]:
    """train 前一半拟合 W（标准化），test 后一半评估 ⇒ (标准化恒等, 线性映射后)。"""
    A, B = _cpu(A), _cpu(B)
    ntr = A.size(0) // 2
    xa, xb = A[:ntr], B[:ntr].clone()
    if derange:                                  # 对照：W 用随机配对拟合
        xb = torch.roll(xb, 1, 0)
    mux, sdx = xa.mean(0), xa.std(0) + 1e-6
    muy, sdy = xb.mean(0), xb.std(0) + 1e-6
    Xtr, Ytr = (xa - mux) / sdx, (xb - muy) / sdy
    Xte, Yte = (A[ntr:] - mux) / sdx, (B[ntr:] - muy) / sdy
    d = Xtr.size(1)
    W = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(d), Xtr.t() @ Ytr)
    return pair_stats(cosmat(Xte, Yte)), pair_stats(cosmat(Xte @ W, Yte))


# --------------------------------------------------------------------------- 阈值 / AUROC
def auroc_rank(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney 口径 AUROC（不依赖 sklearn）。

    ★ 订正 9（复核 P2-15）：**并列取平均秩** —— 原实现用 `argsort` 直接给 1..n 的秩，
    全并列矩阵会返回 **1.0（真值 0.5）**，在退化情形给出"完美区分"的假读数。
    """
    pos, neg = _cpu(pos).flatten(), _cpu(neg).flatten()
    allv = torch.cat([pos, neg])
    n = allv.numel()
    sv, inv = torch.sort(allv)
    starts = torch.nonzero(torch.cat([torch.ones(1, dtype=torch.bool), sv[1:] != sv[:-1]]),
                           as_tuple=False).flatten()
    ends = torch.cat([starts[1:], torch.tensor([n])])
    seg_avg = ((starts + ends - 1).double() / 2.0) + 1.0              # 段内平均秩
    rank_of_sorted = seg_avg[(torch.cumsum(torch.cat([torch.ones(1, dtype=torch.long),
                                                      (sv[1:] != sv[:-1]).long()]), 0) - 1)]
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[inv] = rank_of_sorted
    np_ = pos.numel()
    return float((ranks[:np_].sum() - np_ * (np_ + 1) / 2) / (np_ * neg.numel()))


def youden_threshold(pos: torch.Tensor, neg: torch.Tensor, n_grid: int = 2000) -> dict:
    """最优阈值（Youden J）+ 该点 TPR/FPR。阈值为「>= thr 判同内容」。

    ★ 用 searchsorted 算 CDF，**不构造 [grid, N] 矩阵**（N 可达 4e6，物化会爆内存）。
    """
    pos, neg = _cpu(pos).flatten(), _cpu(neg).flatten()
    ps, _ = torch.sort(pos)
    ns, _ = torch.sort(neg)
    lo = min(float(pos.min()), float(neg.min()))
    hi = max(float(pos.max()), float(neg.max()))
    grid = torch.linspace(lo, hi, n_grid)
    tpr = 1.0 - torch.searchsorted(ps, grid, right=False).float() / pos.numel()
    fpr = 1.0 - torch.searchsorted(ns, grid, right=False).float() / neg.numel()
    j = tpr - fpr
    k = int(torch.argmax(j))
    return dict(thr=float(grid[k]), tpr=float(tpr[k]), fpr=float(fpr[k]), j=float(j[k]))


def score_report(S: torch.Tensor, tag: str) -> dict:
    """一条相似度矩阵的完整读数：配对 Δ±SE + 原始/行中心化两个 AUROC + 最优阈值。"""
    S = _cpu(S)
    N = S.size(0)
    st = pair_stats(S)
    diag = torch.diagonal(S)
    off_mask = ~torch.eye(N, dtype=torch.bool)
    rowmean = (S.sum(1) - diag) / max(N - 1, 1)          # 非对角行均值（= 随机配对期望）
    centered = S - rowmean.unsqueeze(1)                  # 每行去掉"共同分量"
    pos_raw, neg_raw = diag, S[off_mask]
    pos_c, neg_c = centered.diagonal(), centered[off_mask]
    return dict(tag=tag, **{k: v for k, v in st.items() if k != "delta_i"},
                auroc_raw=auroc_rank(pos_raw, neg_raw),
                auroc_centered=auroc_rank(pos_c, neg_c),
                thr_raw=youden_threshold(pos_raw, neg_raw),
                thr_centered=youden_threshold(pos_c, neg_c),
                delta_i=st["delta_i"])


# --------------------------------------------------------------------------- 顺序自检 / 地板
def shuffle_slots(M: torch.Tensor, seed: int) -> torch.Tensor:
    """M3：逐样本独立随机置换 16 个槽（两路用不同 seed ⇒ 置换不同）。"""
    g = torch.Generator().manual_seed(seed)
    M = _cpu(M).clone()
    for i in range(M.size(0)):
        M[i] = M[i][torch.randperm(M.size(1), generator=g)]
    return M


def floor_random(n: int, seed: int, shape=(16, 128)) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, *shape, generator=g)


def floor_bag_of_chars(ids_list: list[list[int]], n_vocab: int = 8192) -> torch.Tensor:
    """地板②：纯词袋（免费规则，含计数类信息）—— 上界参考：它按定义不看跨模态一致性。"""
    B = torch.zeros(len(ids_list), n_vocab)
    for i, ids in enumerate(ids_list):
        for t in ids:
            B[i, t] += 1.0
    return B


def floor_counts(sents: list[str]) -> torch.Tensor:
    """地板③：计数类特征（长度、唯一字数、标点数、汉字数）—— L5 要求的"含计数类"地板。

    ★ 吃**字符串**而不是 token id：char_tokenizer 把汉字映射到 0..8191 的内部 id，
    `chr(id)` 不是原字符（这是本文件早先的一处实现错误）。
    """
    import re
    HAN = re.compile(r"[\u4e00-\u9fff]")
    PUNC = set("，。！？；、：,")
    f = []
    for s in sents:
        f.append([len(s), len(set(s)), sum(1 for c in s if c in PUNC), len(HAN.findall(s))])
    return torch.tensor(f, dtype=torch.float32)


def floor_identity_mediator(ids_list: list[list[int]], imgs: torch.Tensor, seed: int) -> tuple:
    """地板④：恒等中介（不训练）——文本路 = 随机字嵌入的均值；图片路 = 固定随机投影的像素分块。"""
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(8192, 128, generator=g)
    T = torch.stack([emb[torch.tensor(ids, dtype=I64)].mean(0) for ids in ids_list])
    T = T.unsqueeze(1).repeat(1, 16, 1)                           # [N,16,128]
    imgs = _cpu(imgs)
    B = imgs.size(0)
    strips = imgs.reshape(B, 16, -1).mean(-1)                     # [B,16] 像素分块均值
    W = torch.randn(16, 128, generator=g)
    I = strips.unsqueeze(-1) * W                                  # [B,16,128]
    return T, I


def count_tokens_entropy(ids_list: list[list[int]]) -> dict:
    """输入熵估计（按 PREREG §13 的"容量上界 vs 输入熵"要求，报实测口径）。"""
    flat = [t for ids in ids_list for t in ids]
    vals, cnts = np.unique(np.asarray(flat), return_counts=True)
    p = cnts / cnts.sum()
    H = float(-(p * np.log2(p)).sum())
    return dict(n_tokens=len(flat), n_unique=int(vals.size), unigram_bits_per_char=H,
                mean_len=float(np.mean([len(a) for a in ids_list])),
                mean_bits_per_sent=H * float(np.mean([len(a) for a in ids_list])))


def length_matched_delta(S: torch.Tensor, lens) -> dict:
    """★ M8（订正 7）：**同长度、异内容**当负样本的配对 Δ。

    为什么必须做：图片是同一句渲染出来的 ⇒ 图片模因天然与"句长/行数/墨迹量"相关；
    用随机配对（长度分布不同）当对照时，Δ>0 可能只反映"长度相近"。
    ⇒ 只有**同长度下** Δ 仍 ≥2SE 且 ≥0.01，才能说"跨模态相似不是句长造成的"。
    """
    S = _cpu(S)
    lens = [int(x) for x in lens]
    groups: dict[int, list[int]] = {}
    for i, L in enumerate(lens):
        groups.setdefault(L, []).append(i)
    d, no_partner = [], 0
    for i, L in enumerate(lens):
        js = [j for j in groups[L] if j != i]
        if not js:
            no_partner += 1
            continue
        d.append(float(S[i, i] - S[i, js].mean()))
    a = np.asarray(d, float)
    se = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan")
    floor = 0.01
    return dict(n=int(a.size), no_partner=no_partner, delta=float(a.mean()), delta_se=se,
                pass2se=bool(se > 0 and a.mean() >= 2 * se), floor=floor,
                pass_floor=bool(a.mean() >= floor),
                pass_final=bool(se > 0 and a.mean() >= 2 * se and a.mean() >= floor),
                mean_partners=float(np.mean([len(v) - 1 for v in groups.values()])))


__all__ = ["pooled", "cosmat", "slot_cos", "pair_stats", "cka_linear", "cka_perm_null", "linmap",
           "auroc_rank", "youden_threshold", "score_report", "shuffle_slots", "length_matched_delta",
           "floor_random", "floor_bag_of_chars", "floor_counts", "floor_identity_mediator",
           "count_tokens_entropy"]
