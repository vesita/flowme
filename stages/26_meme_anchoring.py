#!/usr/bin/env python3
"""S26 · 判决：模因是「内容表示」还是「表面形式表示」？—— 零训练 · 全程 CPU（跨形式相似性锚定）。

唯一问题：**同一道题（同数字、同答案）、换成不同「表面形式」（模板/措辞/语序/排版/语言）后，
输入卡产出的模因是否相似？** 若显著高于随机配对 ⇒ 模因承载「内容」；否则 ⇒ 模因是表面形式的表示。

设计（对齐后才可比）：
  · 造配对：**同一道题**用 ≥2 种表面形式渲染 ⇒ K 对；三种形式族：
      tpl  = 同数字·同语言(CN)·同算子词·换模板/措辞（排版一致）
      spc  = 同数字·同模板·仅换排版（"a 加 b" vs "a加b"）
      lang = 同数字·跨语言（CN 模板 vs EN 模板 + 换算子词）
    构造方式：数字来自 S14 生成器的 **test 集记录 (a,b,y)**（分布内、答案均匀）；
    模板取自 S14 的 TEMPLATES（本项目自己在用的那套），算子词固定为 CN_OPS/EN_OPS 的第一项
    ⇒ 「同内容」= 同 (a,op,b)，只换外壳。**模板与数字解耦**（14 号脚本 render() 每样本随机抽模板）。
  · 取模因：每条 prompt 过同一个冻结 ckpt（19_*_res_*.pt，residual=True）⇒ M [n,d]；h（模因张量）
    与 h_in（输入卡输出）都存。n_a ≠ n_b 是常态（逐对记录）。
  · ★对齐后比较（三口径全报）：(i) mean-pool 余弦（另报去均值版/末位版）；
    (ii) 线性核 CKA（[K,d] × [K,d]，行=题目）；(iii) 线性映射 W（train 拟合、test 算余弦）。
  · ★对照：随机配对 = 相似度矩阵 C[i,j]=sim(A_i,B_j) 的**非对角全矩阵**（= 均匀随机置换下配对期望）
    ⇒ 报同内容配对 vs 随机配对的分布与**配对 Δ±SE**（Δ_i = C[i,i] − mean_{j≠i} C[i,j]）。
  · 正控：同形式（同模板同排版）**不同数字** ⇒ 相似度应明显更低；若与「不同模板」差不多 ⇒ 度量无区分度、判决作废。

判据（写死）：同内容 Δ ≥ 2×SE(Δ) 且 ≥2 个 ckpt 同向 ⇒ 承载内容 ✓；无显著差异 ⇒ 表面形式表示；
(iii) 显著高于 (i) ⇒ 两条模因在同一线性子空间（可线性对齐）；K<30 ⇒ 样本不足，不下结论。

自检（缺一即无效）：R28（batch=1 vs batch=64 的模因 max|Δ|；本单元无自回归生成）；
零干预恒等（同 prompt 两次前向 max|Δ|）；R29（随机配对必须非零差异 ⇒ 度量非常数）；
格式敏感性正控（同上）。

复用方式（既有源文件零改动、零训练、不重写模型）：exec stages/17_probe_causal_n7.py 的「# ⑥ 主流程」之前
那一段（其内部又 exec stages/14_synth_arith.py 的「# ③ 主循环」之前那一段 = tokenizer + 生成器 + Cards
+ build_batch/meme_rows 等），两个训练主循环都在切片外。S26 只新增：residual 版 hooked logits（可捕获
h 与 h_in）、配对构造、三种相似度口径、CKA/线性映射、逐位置读数。
只允许写：本文件、logs/、/tmp。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★先于 torch 导入：强制关 GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "/home/vesita/coding/my/flowme"
SRC17 = f"{ROOT}/stages/17_probe_causal_n7.py"
TS26 = time.time()

# ============================================================================
# ① 逐字复用 S17 前半段（其内部已 exec S14 前半段 ⇒ 生成器/Cards/meme_rows 全部现成）
# ============================================================================
_s17 = open(SRC17, encoding="utf-8").read()
_c17 = _s17.index("# ⑥ 主流程")
NS17 = {"__name__": "s17_head", "__file__": SRC17}
exec(compile(_s17[:_c17], SRC17, "exec"), NS17)          # noqa: S102
for _k, _v in NS17.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
NS14 = NS17["NS"]                        # S17 内部 exec S14 前半段时用的命名空间（含生成器全量符号）
DATA, TEMPLATES = NS14["DATA"], NS14["TEMPLATES"]
CN_OPS, EN_OPS = NS14["CN_OPS"], NS14["EN_OPS"]
assert NS14["DATA"] is NS17["DATA"], "DATA 应为同一对象（零拷贝复用）"
enc, sha16 = NS17["enc"], NS17["sha16"]
D, FF, NHEAD, PAD_ID = NS17["D"], NS17["FF"], NS17["NHEAD"], NS17["PAD_ID"]
sin_pe, build_batch, meme_rows = NS17["sin_pe"], NS17["build_batch"], NS17["meme_rows"]

# ============================================================================
# ② S26 配置
# ============================================================================
torch.set_num_threads(int(os.environ.get("S26_THREADS", "12")))
SMOKE = os.environ.get("S26_SMOKE") == "1"
K = int(os.environ.get("S26_K", "16" if SMOKE else "128"))
NPERM = int(os.environ.get("S26_NPERM", "8" if SMOKE else "50"))
BATCH = int(os.environ.get("S26_BATCH", "64"))
DEV = "cpu"
assert DEV == "cpu" and torch.cuda.is_available() is False, "必须全程 CPU"
JOBS_ALL = [("add_3d", 1234, 37.25), ("add_3d", 5678, 31.13), ("add_1d", 1234, 96.88)]
JOBS = JOBS_ALL[:1] if SMOKE else JOBS_ALL

print(f"\n[S26] ===== 模因跨形式锚定 | device={DEV} | CUDA_VISIBLE_DEVICES="
      f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} | torch.cuda.is_available()="
      f"{torch.cuda.is_available()} | threads={torch.get_num_threads()} | K={K} NPERM={NPERM} "
      f"batch={BATCH} smoke={SMOKE} =====", flush=True)

# ---- 模板选取（全部取自 S14 的 TEMPLATES，assert 命中）----
CN_PICK = ["请计算 {expr} 等于多少？", "帮我算一下 {expr} 的结果是几？",
           "算 {expr} 是多少？", "麻烦口算 {expr} 得多少？"]
EN_PICK = ["What is {expr}?", "Could you compute {expr} for me",
           "Please solve {expr}.", "How much is {expr}?"]
_tset = {t for t, _cjk in TEMPLATES}
for t in CN_PICK + EN_PICK:
    assert t in _tset, f"模板不在 S14 TEMPLATES 里: {t!r}"
assert all(t in _tset for t in CN_PICK), "CN 模板"
assert all(t in _tset for t in EN_PICK), "EN 模板"
# 顺带记录 S14 生成器的一个退化：裸式模板用普通字符串拼接（"{{expr}} " + t），
# .format() 后残留字面量 "{expr}" ⇒ 该 prompt 里没有任何数字（模型无从作答）。
DEGEN = [t for t, _cjk in TEMPLATES if "{{" in t]
assert not any("{{" in t for t in CN_PICK + EN_PICK), "本单元不使用退化模板"
CN_T, EN_T = CN_PICK, EN_PICK
print(f"[S26] 模板取自 S14 TEMPLATES 共 {len(TEMPLATES)} 句；其中 {len(DEGEN)} 句裸式模板因普通字符串拼接"
      f"（'{{{{expr}}}} ' + t）在 .format() 后残留字面量 '{{expr}}' ⇒ prompt 里没有数字（本单元不使用它们）",
      flush=True)


def op_words(bucket):
    cn = CN_OPS[bucket][0]
    en = EN_OPS[bucket][0]
    return cn, en


def rend(nl_tpl: str, a: int, b: int, opw: str, tight: bool) -> str:
    """与 S14 render() 同构，但模板/算子词/排版由我们指定 ⇒ 同数字可换外壳。"""
    expr = f"{a}{opw}{b}" if tight else f"{a} {opw} {b}"
    out = f"题干：{nl_tpl.format(expr=expr)} → "
    assert "{expr}" not in out, f"模板未被替换: {nl_tpl!r}"
    return out


# ============================================================================
# ③ residual 版 hooked logits：捕获 模因 h（思维卡输出）与 h_in（输入卡输出）
# ============================================================================
CAP: dict = {}
NS17["CAP"] = CAP                          # ★S17 的 meme_rows 在其自身命名空间里查 CAP ⇒ 必须同一对象


def cap26(h, h_in, ids):
    CAP["h"], CAP["h_in"] = h, h_in
    return h


NS17["cap_fn"] = cap26                     # meme_rows 在其自身命名空间里查 cap_fn


def _hooked_logits26(self, ids: torch.Tensor) -> torch.Tensor:
    """与 S19 res 臂 Cards.logits 逐行相同，只多一步：捕获 h 与 h_in（不改任何权重）。"""
    n = ids.size(1)
    m = torch.full((n, n), float("-inf"))
    m = torch.triu(m, diagonal=1)
    m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
    m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
    m = m.repeat_interleave(NHEAD, dim=0)
    pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
    assert n <= pe.size(0)
    x = self.emb(ids) + pe[:n].unsqueeze(0)
    h = self.in_enc(x, src_mask=m)          # 输入卡输出
    h_in = h
    if self.thought is not None:
        h = h + self.thought(h, src_mask=m) if getattr(self, "residual", False) \
            else self.thought(h, src_mask=m)     # ★res 臂：模因张量 = x + Think(x)
    if HOOK["fn"] is not None:
        h = HOOK["fn"](h, h_in, ids)
    return self.head(h)


def ckpt_path(bucket, seed):
    return f"{ROOT}/logs/19_ckpt_{bucket}_res_seed{seed}.pt"


def build_model26(bucket, seed):
    m = Cards(D, FF)
    m.load_state_dict(torch.load(ckpt_path(bucket, seed), map_location="cpu", weights_only=True))
    m.eval()
    m.residual = True
    m.logits = types.MethodType(_hooked_logits26, m)
    return m


@torch.no_grad()
def extract26(model, prompts, bs):
    """与 S17 meme_rows(span='prompt') 同构，只多捕获 h_in。返回 (rows_h, rows_hin, lens)。"""
    model.eval()
    HOOK["fn"] = cap26
    oh, oi, ol = [], [], []
    for i in range(0, len(prompts), bs):
        ch = prompts[i:i + bs]
        lens = [len(p) for p in ch]
        n = max(lens)
        ids = torch.full((len(ch), n), PAD_ID, dtype=torch.long)
        for j, p in enumerate(ch):
            ids[j, :len(p)] = torch.tensor(p)
        model.logits(ids)
        for j in range(len(ch)):
            oh.append(CAP["h"][j, :lens[j]].clone())
            oi.append(CAP["h_in"][j, :lens[j]].clone())
        ol += lens
    HOOK["fn"] = None
    return oh, oi, ol


# ============================================================================
# ④ 三口径相似度
# ============================================================================
def pool(rows, how="mean"):
    return torch.stack([r.mean(0) if how == "mean" else r[-1] for r in rows])


def center(X):
    return X - X.mean(0, keepdim=True)


def cosmat(A, B):
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return A @ B.t()


def pair_stats(C):
    """C[i,j] = sim(A_i, B_j)。返回同内容配对 / 随机配对(全非对角) 与配对 Δ±SE。"""
    Kk = C.size(0)
    d = torch.diagonal(C)
    off = C.clone()
    off.fill_diagonal_(float("nan"))
    row_mean = torch.nanmean(off, dim=1)               # 每个 i 对全部随机伙伴的期望
    delta = d - row_mean
    off_flat = off[~torch.isnan(off)]

    def ms(x):
        n = x.numel()
        return float(x.mean()), (float(x.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan"))
    sm, ss = ms(d)
    om, os_ = ms(off_flat)
    rm, rs = ms(row_mean)
    dm, ds = ms(delta)
    return dict(same=sm, same_se=ss, rand=om, rand_se=os_, rmean=rm, rmean_se=rs,
                d=dm, d_se=ds,
                off_min=float(off_flat.min()), off_max=float(off_flat.max()),
                off_std=float(off_flat.std(unbiased=True)),
                nz_frac=float((delta.abs() > 1e-6).float().mean()), n=Kk)


def cka_linear(X, Y):
    X, Y = center(X), center(Y)
    return float((X.t() @ Y).norm() ** 2 / ((X.t() @ X).norm() * (Y.t() @ Y).norm()).clamp_min(1e-30))


def linmap(A, B, lam=1e-2, derange=False):
    """train 前一半拟合 W（标准化后），test 后一半评估：返回 (i)_same, (iii)_same, Δ 与随机版。"""
    ntr = A.size(0) // 2
    xa_tr, xb_tr, xa_te, xb_te = A[:ntr], B[:ntr], A[ntr:], B[ntr:]
    if derange:                                  # 对照：W 用「随机配对」的 train 拟合
        xb_tr = torch.roll(xb_tr, 1, 0)
    mux, sdx = xa_tr.mean(0), xa_tr.std(0) + 1e-6
    muy, sdy = xb_tr.mean(0), xb_tr.std(0) + 1e-6
    Xtr, Ytr = (xa_tr - mux) / sdx, (xb_tr - muy) / sdy
    Xte, Yte = (xa_te - mux) / sdx, (xb_te - muy) / sdy
    dd = Xtr.size(1)
    W = torch.linalg.solve(Xtr.t() @ Xtr + lam * torch.eye(dd), Xtr.t() @ Ytr)
    st_i = pair_stats(cosmat(Xte, Yte))          # (i) 同 test 行
    st_iii = pair_stats(cosmat(Xte @ W, Yte))    # (iii) 线性映射后
    return st_i, st_iii


def pr_eff(X):
    """参与率 PR = (Σλ)²/Σλ²（有效维度），λ 为去均值后协方差的特征值。"""
    Xc = center(X)
    s2 = torch.linalg.svdvals(Xc) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def pos_var_profile(rows):
    """逐位置方差：v[p] = mean_j Var_i(M[i,p,j])。"""
    n = max(r.size(0) for r in rows)
    v = []
    for p in range(n):
        sel = [r[p] for r in rows if r.size(0) > p]
        v.append(float(torch.stack(sel).var(0, unbiased=True).mean()) if len(sel) > 1 else float("nan"))
    return v


def pos_diff_profile(rowsA, rowsB, align="start"):
    """对齐后的逐位置差异能量 d[p] = mean_{i,j}(M_A[i,p,j]-M_B[i,p,j])²。"""
    nmax = max(max(r.size(0) for r in rowsA), max(r.size(0) for r in rowsB))
    s = [0.0] * nmax
    c = [0] * nmax
    for ra, rb in zip(rowsA, rowsB):
        m = min(ra.size(0), rb.size(0))
        for p in range(m):
            ia = p if align == "start" else ra.size(0) - 1 - p
            ib = p if align == "start" else rb.size(0) - 1 - p
            s[p] += float(((ra[ia] - rb[ib]) ** 2).mean())
            c[p] += 1
    return [(s[p] / c[p] if c[p] else float("nan")) for p in range(nmax)], c


def conc(d, q=0.2):
    """差异能量集中度：能量最大的前 q 比例位置占总能量的份额。"""
    xs = np.array([x for x in d if x == x], float)
    if xs.size == 0 or xs.sum() <= 0:
        return float("nan"), -1
    o = np.argsort(-xs)
    kk = max(1, int(math.ceil(q * xs.size)))
    return float(xs[o[:kk]].sum() / xs.sum()), int(o[0])


# ============================================================================
# ⑤ 主流程：每个 ckpt × {tpl, spc, lang} × 三口径
# ============================================================================
RES: dict = {}
rng = np.random.default_rng(20261008)
PERM = [torch.tensor(rng.permutation(K)) for _ in range(NPERM)]
SUMMARY: list[dict] = []

for bucket, seed, em_anchor in JOBS:
    t0 = time.time()
    key = f"{bucket}@res@{seed}"
    model = build_model26(bucket, seed)
    ck = ckpt_path(bucket, seed)
    cnw, enw = op_words(bucket)
    recs = DATA[bucket]["test"][:K]
    assert len(recs) == K, f"{bucket} test 不足 {K}"
    QA = [(int(r["a"]), int(r["b"]), int(r["y"])) for r in recs]
    print(f"\n[S26] ===== ckpt={key} {os.path.basename(ck)} sha16={sha16(ck)} | bucket={bucket} "
          f"| K={K} 题（test 前 {K} 条，(a,b,y) 直接取自 S14 生成器）| S19 锚 EM={em_anchor}% "
          f"| CN算子词={cnw!r} EN算子词={enw!r} =====", flush=True)

    # ---- 自检1/2：R28（batch 依赖）+ 零干预恒等（同 prompt 两次前向）----
    smoke_p = [enc(rend(CN_T[i % 4], QA[i][0], QA[i][1], cnw, False)) for i in range(min(16, K))]
    h1a, i1a, l1a = extract26(model, smoke_p, 1)
    h1b, i1b, _ = extract26(model, smoke_p, 1)
    id_h = max(float((a - b).abs().max()) for a, b in zip(h1a, h1b))
    id_i = max(float((a - b).abs().max()) for a, b in zip(i1a, i1b))
    h64, i64, _ = extract26(model, smoke_p, BATCH)
    r28_h = max(float((a - b).abs().max()) for a, b in zip(h1a, h64))
    r28_i = max(float((a - b).abs().max()) for a, b in zip(i1a, i64))
    # 与 S17 meme_rows 的互检（同一 ckpt、同一 prompt、同一 h 口径）
    mr = meme_rows(model, [dict(p=p) for p in smoke_p], "prompt")
    xchk = max(float((a - b).abs().max()) for a, b in zip(h1a, mr))
    rms_h = float(torch.cat([r.reshape(-1) for r in h1a]).pow(2).mean().sqrt())
    rms_i = float(torch.cat([r.reshape(-1) for r in i1a]).pow(2).mean().sqrt())
    print(f"[SELF1-R28] {key} batch=1 vs batch={BATCH}（n={len(smoke_p)} prompt，纯前向无生成）："
          f"模因 max|Δ|={r28_h:.3e}（RMS|h|={rms_h:.3f} ⇒ 相对={r28_h/rms_h:.2e}）输入卡 "
          f"max|Δ|={r28_i:.3e}（RMS={rms_i:.3f} ⇒ 相对={r28_i/rms_i:.2e}）⇒ "
          f"{'逐位相同（无 batch 依赖）' if max(r28_h, r28_i) == 0.0 else '★有微小 batch 依赖（浮点归约序）⇒ 全流程只用同一 batch 口径'}"
          f" | 本单元无自回归生成（只前向取模因）", flush=True)
    print(f"[SELF2-恒等] {key} 同 prompt 两次前向：模因 max|Δ|={id_h:.3e} 输入卡 max|Δ|={id_i:.3e} ⇒ "
          f"{'精确为 0 ⇒ 通过' if max(id_h, id_i) == 0.0 else '★非 0 ⇒ 口径有误'}"
          f" | 与 S17 meme_rows 互检 max|Δ|={xchk:.3e}", flush=True)

    # ---- 素材：每族每模板的模因 ----
    texts = {}
    for ti, t in enumerate(CN_T):
        texts[("cn", ti, False)] = [enc(rend(t, a, b, cnw, False)) for a, b, _ in QA]
        texts[("cn", ti, True)] = [enc(rend(t, a, b, cnw, True)) for a, b, _ in QA]
    for ti, t in enumerate(EN_T):
        texts[("en", ti, False)] = [enc(rend(t, a, b, enw, False)) for a, b, _ in QA]
    rowsH, rowsI, LN = {}, {}, {}
    for kk, ps in texts.items():
        rh, ri, ln = extract26(model, ps, BATCH)
        rowsH[kk], rowsI[kk], LN[kk] = rh, ri, ln
    n_lens = [r.size(0) for r in rowsH[("cn", 0, False)]]
    neq = sum(1 for ti in range(4) for i in range(K)
              if LN[("cn", ti, False)][i] != LN[("cn", (ti + 1) % 4, False)][i])
    print(f"[DATA] {key} prompt token 数 CN={min(n_lens)}..{max(n_lens)} | "
          f"CN 模板两两不同长度占比={neq}/{4*K}={100.0*neq/(4*K):.0f}% ⇒ n_a≠n_b 是常态 | "
          f"素材 = 4×CN(松) + 4×CN(紧) + 4×EN = 12 组 × {K} 题", flush=True)

    # ---- 逐族 × 三口径 ----
    fams: dict = {}

    def run_family(name, pairs, rowsX, rowsY):
        """pairs: [(label, keyX, keyY)]；返回该族逐 pair 的三口径与正控。"""
        out = []
        for lab, kx, ky in pairs:
            X, Y = pool(rowsX[kx]), pool(rowsY[ky])
            sa = pair_stats(cosmat(X, Y))                            # (i) 原始 mean-pool
            sb = pair_stats(cosmat(center(X), center(Y)))            # (i') 去均值 mean-pool
            Xl, Yl = pool(rowsX[kx], "last"), pool(rowsY[ky], "last")
            sc = pair_stats(cosmat(Xl, Yl))                          # (i'') 末位
            ck_al = cka_linear(X, Y)                                 # (ii) CKA 对齐
            ck_pe = [cka_linear(X, Y[p]) for p in PERM]              # (ii) CKA 随机行置换
            cm, cs = float(np.mean(ck_pe)), float(np.std(ck_pe, ddof=1) / math.sqrt(len(ck_pe)))
            st_i, st_iii = linmap(X, Y)                              # (iii) 线性映射
            _, st_iii_r = linmap(X, Y, derange=True)                 # 对照：W 用随机配对拟合
            out.append(dict(label=lab, i=sa, ic=sb, il=sc,
                            cka_al=ck_al, cka_perm=cm, cka_perm_se=cs, cka_d=ck_al - cm,
                            lin_i=st_i, lin_iii=st_iii, lin_iii_rand=st_iii_r,
                            gain=st_iii["d"] - st_i["d"],
                            same_gain=st_iii["same"] - st_i["same"]))
        return out

    tpl_pairs = [(f"CN{t1}/CN{t2}", ("cn", t1, False), ("cn", t2, False))
                 for t1 in range(4) for t2 in range(t1 + 1, 4)]
    spc_pairs = [(f"CN{t}/松紧", ("cn", t, False), ("cn", t, True)) for t in range(4)]
    lang_pairs = [(f"CN{tc}/EN{te}", ("cn", tc, False), ("en", te, False))
                  for tc in range(4) for te in range(4)]
    fams["tpl"] = run_family("tpl", tpl_pairs, rowsH, rowsH)
    fams["spc"] = run_family("spc", spc_pairs, rowsH, rowsH)
    fams["lang"] = run_family("lang", lang_pairs, rowsH, rowsH)
    # 正控：同形式（同模板同排版）不同数字 ⇒ 相似度矩阵的非对角
    ctrl = []
    for t in range(4):
        X = pool(rowsH[("cn", t, False)])
        C = cosmat(X, X).clone()
        C.fill_diagonal_(float("nan"))
        off = C[~torch.isnan(C)]
        ctrl.append(dict(t=t, same_form_diff_num=float(off.mean()),
                         se=float(off.std(unbiased=True) / math.sqrt(off.numel()))))
    # 输入卡输出（h_in）上重复主族的口径 (i)
    fams_in = {}
    for nm, prs in (("tpl", tpl_pairs), ("spc", spc_pairs), ("lang", lang_pairs)):
        fams_in[nm] = run_family(nm, prs, rowsI, rowsI)

    # ---- 打印 ----
    def agg(fam):
        ds = [p["i"]["d"] for p in fams[fam]]
        ses = [p["i"]["d_se"] for p in fams[fam]]
        return (float(np.mean(ds)), float(np.mean(ses)), min(ds), max(ds),
                float(np.mean([p["i"]["same"] for p in fams[fam]])),
                float(np.mean([p["i"]["rand"] for p in fams[fam]])),
                float(np.mean([p["cka_d"] for p in fams[fam]])),
                float(np.mean([p["gain"] for p in fams[fam]])))
    for fam in ("tpl", "spc", "lang"):
        a = agg(fam)
        fs = fams[fam]
        lead = fs[0]
        print(f"[S26-{fam}] {key} pairs={len(fs)} | (i) mean-pool cos: 同内容={a[4]:.4f} "
              f"随机={a[5]:.4f} Δ={a[0]:+.4f}±{a[1]:.4f} (逐pair Δ {a[2]:+.4f}..{a[3]:+.4f}, "
              f"2SE 判据 {'过' if a[0] >= 2*a[1] else '★不过'}) | (i') 去均值 Δ="
              f"{np.mean([p['ic']['d'] for p in fs]):+.4f} | (i'') 末位 Δ="
              f"{np.mean([p['il']['d'] for p in fs]):+.4f} | (ii) CKA 对齐={lead['cka_al']:.4f} "
              f"随机={lead['cka_perm']:.4f}±{lead['cka_perm_se']:.4f} Δ均={a[6]:+.4f} | "
              f"(iii) 标准化恒等 Δ均={np.mean([p['lin_i']['d'] for p in fs]):+.4f} "
              f"(same={np.mean([p['lin_i']['same'] for p in fs]):.4f}) ⇒ 映射后 Δ均="
              f"{np.mean([p['lin_iii']['d'] for p in fs]):+.4f} "
              f"(same={np.mean([p['lin_iii']['same'] for p in fs]):.4f}) "
              f"增益Δ={np.mean([p['gain'] for p in fs]):+.4f} 增益same="
              f"{np.mean([p['same_gain'] for p in fs]):+.4f} | (iii)对照(W用随机pair拟合) Δ均="
              f"{np.mean([p['lin_iii_rand']['d'] for p in fs]):+.4f}", flush=True)
    p0 = fams["tpl"][0]["i"]
    print(f"[SELF3-R29] {key} 随机配对非零性：非对角相似度 范围={p0['off_min']:.4f}..{p0['off_max']:.4f} "
          f"std={p0['off_std']:.4f} | 配对 |Δ|>1e-6 的比例={p0['nz_frac']*100:.1f}% ⇒ "
          f"{'度量有区分度（非常数）' if p0['off_std'] > 1e-6 and p0['nz_frac'] > 0.9 else '★近常数 ⇒ 度量无区分度'}",
          flush=True)
    cm_ = float(np.mean([c["same_form_diff_num"] for c in ctrl]))
    cs_ = float(np.mean([c["se"] for c in ctrl]))
    a_tpl = agg("tpl")
    a_spc = agg("spc")
    print(f"[SELF4-正控] {key} 同形式(同模板同排版)不同数字 cos={cm_:.4f}±{cs_:.4f} vs "
          f"不同模板同数字 cos={a_tpl[4]:.4f}："
          f"{'不同数字明显更低 ⇒ 度量对内容敏感、判决有效' if cm_ < a_tpl[4] - 2*a_tpl[1] else '★两者相当 ⇒ 度量无区分度（判决作废）'}"
          f" | 参考：不同排版同数字 cos={a_spc[4]:.4f}", flush=True)

    # ---- 辅助读数：PR + 逐位置方差 ----
    XA = pool(rowsH[("cn", 0, False)])
    XB = pool(rowsH[("cn", 1, False)])
    XAB = torch.cat([XA, XB], 0)
    prA, prB, prAB = pr_eff(XA), pr_eff(XB), pr_eff(XAB)
    vA = pos_var_profile(rowsH[("cn", 0, False)])
    vAf = np.array([x for x in vA if x == x], float)
    small = float((vAf < 0.1 * vAf.max()).mean())
    dspc, _ = pos_diff_profile(rowsH[("cn", 0, False)], rowsH[("cn", 0, True)], "start")
    dtpl, _ = pos_diff_profile(rowsH[("cn", 0, False)], rowsH[("cn", 1, False)], "end")
    cspc, ispc = conc(dspc)
    ctpl, itpl = conc(dtpl)
    print(f"[AUX] {key} PR(有效维度, mean-pool [K,d]): 形式A={prA:.1f} 形式B={prB:.1f} 合并={prAB:.1f} "
          f"(d={D}, K={K}) | 逐位置方差 均值={vAf.mean():.4f} 最大@{int(np.argmax(vAf))} "
          f"最小={vAf.min():.2e} | 方差 <10%max 的位置占 {small*100:.0f}%", flush=True)
    print(f"[AUX] {key} 逐位置差异能量集中度：spc(松/紧, 从头对齐) 前20%位置占 {cspc*100:.0f}%"
          f"(峰@p{ispc}) | tpl(CN0/CN1, 从尾对齐) 前20%位置占 {ctpl*100:.0f}%(峰@p{itpl}, "
          f"对齐尾) ⇒ {'差异集中在少数位置' if max(cspc, ctpl) > 0.4 else '差异较均匀铺开'}", flush=True)

    RES[key] = dict(bucket=bucket, seed=seed, em_anchor=em_anchor, sha16=sha16(ck), K=K,
                    self1=dict(r28_h=r28_h, r28_i=r28_i, batch=BATCH),
                    self2=dict(id_h=id_h, id_i=id_i, xchk_s17=xchk),
                    fams={f: fams[f] for f in fams}, fams_in={f: fams_in[f] for f in fams_in},
                    ctrl=ctrl, pr=dict(A=prA, B=prB, AB=prAB),
                    posvar=dict(mean=float(vAf.mean()), max=float(vAf.max()),
                                argmax=int(np.argmax(vAf)), small_frac=small),
                    posdiff=dict(spc_conc=cspc, spc_peak=ispc, tpl_conc=ctpl, tpl_peak=itpl),
                    wall=time.time() - t0)
    print(f"[TIME] {key} 用时 {(time.time()-t0)/60:.2f} min", flush=True)
    del model

# ============================================================================
# ⑥ 总判决
# ============================================================================
print("\n[S26-SUMMARY] ===== 三口径 · 同内容 vs 随机配对（每 ckpt 每族，mean-pool cos 主口径）=====")
print("[S26-SUMMARY] ckpt | 族 | (i) 同内容 | (i) 随机 | (i) Δ±SE | 过2SE | (ii) CKA Δ | "
      "(iii) Δ | (iii)-(i) 增益 | 正控(同形式异数字)", flush=True)
for key, R in RES.items():
    for fam in ("tpl", "spc", "lang"):
        fs = R["fams"][fam]
        d = float(np.mean([p["i"]["d"] for p in fs]))
        se = float(np.mean([p["i"]["d_se"] for p in fs]))
        ckad = float(np.mean([p["cka_d"] for p in fs]))
        gain = float(np.mean([p["gain"] for p in fs]))
        iii = float(np.mean([p["lin_iii"]["d"] for p in fs]))
        print(f"[S26-SUMMARY] {key:18s} | {fam:4s} | {np.mean([p['i']['same'] for p in fs]):.4f} | "
              f"{np.mean([p['i']['rand'] for p in fs]):.4f} | {d:+.4f}±{se:.4f} | "
              f"{'✓' if d >= 2*se else '✗'} | {ckad:+.4f} | {iii:+.4f} | {gain:+.4f} | "
              f"{np.mean([c['same_form_diff_num'] for c in R['ctrl']]):.4f}", flush=True)
print("[S26-SUMMARY] 输入卡输出(h_in)上的 (i) 口径 Δ±SE（对照模因 h）：", flush=True)
for key, R in RES.items():
    print("[S26-SUMMARY] " + key + " | " + " | ".join(
        f"{fam}: {float(np.mean([p['i']['d'] for p in R['fams_in'][fam]])):+.4f}"
        f"±{float(np.mean([p['i']['d_se'] for p in R['fams_in'][fam]])):.4f}"
        for fam in ("tpl", "spc", "lang")), flush=True)

# ---- 判据 ----
if K < 30:
    V = f"★K={K}<30 ⇒ 样本不足，不下结论（不许硬选）"
else:
    ok = {fam: [bool(np.mean([p["i"]["d"] for p in R["fams"][fam]]) >=
                    2 * np.mean([p["i"]["d_se"] for p in R["fams"][fam]])) for R in RES.values()]
          for fam in ("tpl", "spc", "lang")}
    ctrl_ok = all(min(c["same_form_diff_num"] for c in R["ctrl"]) <
                  np.mean([p["i"]["same"] for p in R["fams"]["tpl"]]) -
                  2 * np.mean([p["i"]["d_se"] for p in R["fams"]["tpl"]]) for R in RES.values())
    sign_ok = all(all(p["i"]["d"] > 0 for p in R["fams"]["tpl"]) for R in RES.values())
    n_ck = len(RES)
    if not ctrl_ok:
        V = "★判据作废：格式敏感性正控未通过（不同数字与不同模板的相似度相当 ⇒ 度量无区分度）"
    elif all(ok["tpl"]) and n_ck >= 2 and sign_ok:
        V = (f"★模因承载内容（锚定成功）：换模板 Δ>0 且 ≥2×SE，{n_ck} 个 ckpt 同向；"
             f"族一致={{{', '.join(f'{f}:{all(ok[f])}' for f in ok)}}}")
    elif not any(ok["tpl"]):
        V = "★模因是表面形式的表示（换模板后相似度不高于随机配对，如实报）"
    else:
        V = f"★部分成立/方向不稳：逐族过 2SE = {ok}（n_ck={n_ck}）⇒ 如实报，不硬选"
gain_ok = all(float(np.mean([p["same_gain"] for p in R["fams"]["tpl"]])) > 0.005 for R in RES.values())
print(f"\n[S26-VERDICT] {V}", flush=True)
print(f"[S26-VERDICT] (iii) 线性映射后同内容余弦 − 标准化恒等同内容余弦（tpl 族逐 ckpt "
      f"{[round(float(np.mean([p['same_gain'] for p in R['fams']['tpl']])), 4) for R in RES.values()]}）"
      f"⇒ {'两条模因基本在同一线性子空间（映射后同内容相似度更高）' if gain_ok else '线性映射未提高同内容相似度（差异非单纯线性可消）'}",
      flush=True)
print(f"[S26-VERDICT] PR: " + " | ".join(
    f"{k}: A={R['pr']['A']:.1f} B={R['pr']['B']:.1f} 合并={R['pr']['AB']:.1f}" for k, R in RES.items()),
    flush=True)
print(f"[S26-VERDICT] ckpt sha16: " + " | ".join(f"{k}={R['sha16'][:16]}" for k, R in RES.items()),
      flush=True)
print(f"[META] device={DEV} cuda_is_available={torch.cuda.is_available()} threads="
      f"{torch.get_num_threads()} K={K} NPERM={NPERM} ckpt数={len(RES)} | 墙钟="
      f"{(time.time()-TS26)/60:.2f}min", flush=True)
with open(f"{ROOT}/logs/26_results.json", "w", encoding="utf-8") as f:
    json.dump(dict(verdict=V, K=K, nperm=NPERM, res={k: RES[k] for k in RES},
                   wall_min=(time.time() - TS26) / 60), f, ensure_ascii=False, indent=1)
print("[DONE] exit=0", flush=True)
